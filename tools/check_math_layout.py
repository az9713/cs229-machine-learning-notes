#!/usr/bin/env python3
"""Check inline and display MathJax layout in a generated HTML reader."""

import argparse
import sys
from pathlib import Path

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import sync_playwright

MEASURE_LAYOUT = """
() => {
    const formulas = [...document.querySelectorAll('mjx-container')];
    const boxes = new Map();
    for (const formula of formulas) {
        const inline = formula.getAttribute('display') !== 'true';
        boxes.set(formula, inline);
        const wrapper = formula.parentElement.closest(
            'span.math, .inline-math, .math-inline, .equation, .math-display, .math-scroll'
        );
        if (wrapper) boxes.set(wrapper, inline);
    }

    let unhiddenInlineScrollers = 0;
    let wideInline = 0;
    let wideDisplay = 0;
    let blockedOverflow = 0;
    for (const [box, inline] of boxes) {
        if (!box.clientWidth) continue;
        const style = getComputedStyle(box);
        const scrolls = ['auto', 'scroll'].includes(style.overflowX);
        const hidden = style.scrollbarWidth === 'none'
            || getComputedStyle(box, '::-webkit-scrollbar').display === 'none';
        const scrollsVertically = ['auto', 'scroll'].includes(style.overflowY);
        if (inline && !hidden && (scrolls || scrollsVertically)) {
            unhiddenInlineScrollers++;
        }
        if (box.scrollWidth - box.clientWidth <= 1) continue;
        // Overflow can reach an outer equation wrapper when this box leaves it visible.
        const outer = box.parentElement.closest('.equation, .math-display, .math-scroll, span.math, .inline-math, .math-inline');
        if (style.overflowX === 'visible' && boxes.has(outer)) continue;
        if (inline) wideInline++;
        else wideDisplay++;
        const original = box.scrollLeft;
        box.scrollLeft = 24;
        if (box.scrollLeft === original) box.scrollLeft = -24;
        if (!scrolls || box.scrollLeft === original) blockedOverflow++;
        box.scrollLeft = original;
    }
    return {
        rendered: formulas.length,
        mathErrors: document.querySelectorAll('mjx-merror').length,
        unhiddenInlineScrollers, wideInline, wideDisplay, blockedOverflow,
        pageOverflow: document.documentElement.scrollWidth > innerWidth + 1,
    };
}
"""


def check_page(page, target):
    response = page.goto(target, wait_until="load")
    if response and not response.ok:
        raise RuntimeError(f"HTTP {response.status}")
    page.wait_for_selector("mjx-container", state="attached", timeout=15000)
    page.evaluate("() => MathJax.startup.promise.then(() => undefined)")
    return page.evaluate(MEASURE_LAYOUT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pages", nargs="+", help="Local HTML files or deployed HTTP(S) URLs")
    parser.add_argument("--chrome", help="Path to an installed Chromium or Edge executable")
    args = parser.parse_args()
    targets = [
        name if name.startswith(("http://", "https://"))
        else Path(name).resolve(strict=True).as_uri()
        for name in args.pages
    ]

    failures = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path=args.chrome,
            ignore_default_args=["--hide-scrollbars"],
        )
        for width, height in ((1440, 900), (390, 844)):
            page = browser.new_page(viewport={"width": width, "height": height})
            for target in targets:
                try:
                    result = check_page(page, target)
                except (RuntimeError, PlaywrightError) as error:
                    failures.append(f"{target} at {width}px: {error}")
                    continue
                print(f"{target} at {width}px: {result}")
                if (
                    not result["rendered"]
                    or result["mathErrors"]
                    or result["unhiddenInlineScrollers"]
                    or result["blockedOverflow"]
                    or result["pageOverflow"]
                ):
                    failures.append(f"{target} at {width}px: {result}")
            page.close()
        browser.close()

    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    print(f"Passed {len(targets) * 2} desktop and mobile math layout checks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
