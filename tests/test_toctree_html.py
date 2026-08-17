"""Tests for the BeautifulSoup-free toctree rewriter."""

import pytest

from bs4 import BeautifulSoup

from pydata_sphinx_theme._toctree_html import (
    build_template,
    render_page_toc,
    render_toctree,
    rewrite_page_toc,
    rewrite_toctree,
)
from pydata_sphinx_theme.toctree import add_collapse_checkboxes


def bs4_reference(html: str, kind: str, show_nav_level: int) -> str:
    """Rewrite a toctree fragment the way the theme did with BeautifulSoup.

    Kept as a reference implementation so that
    :func:`pydata_sphinx_theme._toctree_html.rewrite_toctree` can be checked
    against it; the theme itself no longer runs this code.
    """
    soup = BeautifulSoup(html, "html.parser")
    for li in soup("li", {"class": "current"}):
        li["class"].append("active")
    for li in soup.select("li"):
        if li.find("a"):
            href = li.find("a")["href"]
            if "#" in href and href != "#":
                li.decompose()
    if kind == "sidebar":
        for ul in soup("ul", recursive=False):
            ul.attrs["class"] = [*ul.attrs.get("class", []), "nav", "bd-sidenav"]
        if show_nav_level == 0:
            partcaptions = soup.find_all("p", attrs={"class": "caption"})
            if len(partcaptions):
                new_soup = BeautifulSoup(
                    "<ul class='list-caption'></ul>", "html.parser"
                )
                for caption in partcaptions:
                    for sibling in caption.next_siblings:
                        if sibling.name == "ul":
                            toclist = sibling
                            break
                    li = soup.new_tag("li", attrs={"class": "toctree-l0"})
                    li.extend([caption, toclist])
                    new_soup.ul.append(li)
                soup = new_soup
        add_collapse_checkboxes(soup)
        for ii in range(show_nav_level):
            for details in soup.select(f"li.toctree-l{ii} > details"):
                details["open"] = "open"
    return str(soup)


CAPTION = '<p aria-level="2" class="caption" role="heading"><span class="caption-text">Part &amp; parcel</span></p>'  # noqa: E501


def _entry(level, href, title, *, current=False, children=""):
    classes = f"toctree-l{level} current" if current else f"toctree-l{level}"
    link = "current reference internal" if current else "reference internal"
    return (
        f'<li class="{classes}"><a class="{link}" href="{href}">{title}</a>'
        f"{children}</li>\n"
    )


SIMPLE = (
    "<ul>\n"
    + _entry(1, "a.html", "Page A")
    + _entry(
        1,
        "b.html",
        "Page B",
        current=True,
        children='<ul class="current">\n'
        + _entry(2, "#", "Page B", current=True)
        + _entry(2, "b2.html", "Page B2")
        + "</ul>\n",
    )
    + _entry(1, "https://example.com", "External")
    + "</ul>\n"
)
WITH_ANCHORS = (
    "<ul>\n"
    + _entry(
        1,
        "#",
        "Self",
        current=True,
        children="<ul>\n" + _entry(2, "#section", "A section") + "</ul>\n",
    )
    + "</ul>\n"
)
WITH_PARTS = (
    CAPTION
    + "<ul>\n"
    + _entry(1, "a.html", "Page A")
    + _entry(
        1,
        "b.html",
        "Page B",
        current=True,
        children='<ul class="current">\n'
        + _entry(2, "#", "Page B", current=True)
        + "</ul>\n",
    )
    + "</ul>\n"
    + CAPTION
    + "<ul>\n"
    + _entry(1, "c.html", "Page C")
    + "</ul>\n"
)
DEEP = (
    "<ul>\n"
    + _entry(
        1,
        "a.html",
        "Page A",
        children="<ul>\n"
        + _entry(
            2,
            "b.html",
            "Page B",
            children="<ul>\n" + _entry(3, "c.html", "C") + "</ul>\n",
        )
        + "</ul>\n",
    )
    + "</ul>\n"
)
# titles with markup and characters that entity-escaping has to round-trip
TRICKY = (
    "<ul>\n"
    '<li class="toctree-l1"><a class="reference internal" href="x.html">'
    'a &amp; b &lt;c&gt; d &#64; e <code class="docutils literal notranslate">'
    "<span>f</span></code> <em>g</em><br/></a></li>\n"
    '<li class="toctree-l1"><a class="reference internal" href=\'q"uote.html\' '
    'title="say &quot;hi&quot;">Quoted</a></li>\n'
    "</ul>\n"
)

FRAGMENTS = {
    "simple": SIMPLE,
    "with_anchors": WITH_ANCHORS,
    "with_parts": WITH_PARTS,
    "deep": DEEP,
    "tricky": TRICKY,
    "empty": "",
}


@pytest.mark.parametrize("name", sorted(FRAGMENTS))
@pytest.mark.parametrize("kind", ["sidebar", "raw"])
@pytest.mark.parametrize("show_nav_level", [0, 1, 2])
def test_matches_beautifulsoup(name, kind, show_nav_level) -> None:
    """The rewriter must reproduce the old BeautifulSoup pipeline byte for byte."""
    html = FRAGMENTS[name]
    expected = bs4_reference(html, kind, show_nav_level)
    root = rewrite_toctree(html, kind=kind, show_nav_level=show_nav_level)
    assert render_toctree(root) == expected


@pytest.mark.parametrize(
    "html",
    [
        "plain text & more",
        "<span>a &amp; b &lt; c &#64; d &nbsp; e</span>",
        '<a href="x?a=1&b=2" title=\'he said "hi"\'>x</a>',
        "<a title='mixed \"quotes\" and &#39;apostrophes&#39;'>x</a>",
        "<ul><li>a<br>b<img src='x.png' alt='y'></li></ul>",
        '<p class="  a   b  " id="z">t</p>',
        "<!-- a comment --><ul></ul>",
        "<ul><li>unclosed",
    ],
)
def test_serializer_matches_beautifulsoup(html) -> None:
    """Serialization must match BeautifulSoup's default formatter byte for byte."""
    root = rewrite_toctree(html, kind="raw", show_nav_level=1)
    assert render_toctree(root) == str(BeautifulSoup(html, "html.parser"))


# A toctree of five pages in one directory, three of them nested, split over two
# parts. Rendering it for each page in turn is what Sphinx does on a real site.
PART1 = [("a.html", "Page A", []), ("b.html", "Page B", [("b1.html", "B1", [])])]
PART2 = [("c.html", "Page C", [("c1.html", "C1", []), ("c2.html", "C2", [])])]
ALL_HREFS = ["a.html", "b.html", "b1.html", "c.html", "c1.html", "c2.html"]


def _ancestors(tree, current, chain=()):
    for href, _, children in tree:
        if href == current:
            return chain
        found = _ancestors(children, current, (*chain, href))
        if found is not None:
            return found
    return None


def _entries(tree, current, ancestors, level):
    out = []
    for href, title, children in tree:
        is_current = href == current
        on_chain = is_current or href in ancestors
        inner = ""
        if children:
            marker = ' class="current"' if href in ancestors else ""
            inner = (
                f"<ul{marker}>\n"
                + _entries(children, current, ancestors, level + 1)
                + "</ul>\n"
            )
        li_class = f"toctree-l{level} current" if on_chain else f"toctree-l{level}"
        link = "current reference internal" if is_current else "reference internal"
        out.append(
            f'<li class="{li_class}"><a class="{link}" '
            f'href="{"#" if is_current else href}">{title}</a>{inner}</li>\n'
        )
    return "".join(out)


def sphinx_fragment(current: str, parts: bool) -> str:
    """Return the toctree fragment Sphinx would render on page ``current``."""
    out = []
    for part in (PART1, PART2):
        ancestors = _ancestors(part, current)
        if parts:
            out.append(CAPTION)
        marker = ' class="current"' if ancestors is not None else ""
        out.append(f"<ul{marker}>\n")
        out.append(_entries(part, current, ancestors or (), 1))
        out.append("</ul>\n")
    return "".join(out)


@pytest.mark.parametrize("parts", [False, True])
@pytest.mark.parametrize("show_nav_level", [0, 1, 2])
def test_template_moves_current_markers(parts, show_nav_level) -> None:
    """Re-targeting a cached template must equal a fresh rewrite for that page."""
    built_for = "b1.html"
    root = rewrite_toctree(
        sphinx_fragment(built_for, parts), kind="sidebar", show_nav_level=show_nav_level
    )
    template = build_template(root, show_nav_level=show_nav_level, self_href=built_for)
    # the template must reproduce the page it was built for (this is what
    # generate_toctree_html checks before caching it)
    assert template.render(built_for, built_for) == render_toctree(root)
    assert template.render("", built_for) == render_toctree(root)
    for href in ALL_HREFS:
        expected = render_toctree(
            rewrite_toctree(
                sphinx_fragment(href, parts),
                kind="sidebar",
                show_nav_level=show_nav_level,
            )
        )
        assert template.render(href, built_for) == expected, href
    assert template.render("not-an-entry.html", built_for) is None


def bs4_page_toc_reference(html: str, show_toc_level: int, kind: str):
    """Rewrite the in-page TOC the way the theme did with BeautifulSoup."""
    soup = BeautifulSoup(html, "html.parser")

    def add_header_level_recursive(ul, level):
        if ul is None:
            return
        if level <= (show_toc_level + 1):
            ul["class"] = [*ul.get("class", []), "pst-show_toc_level"]
        for li in ul("li", recursive=False):
            li["class"] = [*li.get("class", []), f"toc-h{level}"]
            add_header_level_recursive(li.find("ul", recursive=False), level + 1)

    add_header_level_recursive(soup.find("ul"), 1)
    for ul in soup("ul"):
        ul["class"] = [*ul.get("class", []), "nav", "section-nav", "flex-column"]
    for li in soup("li"):
        li["class"] = [*li.get("class", []), "nav-item", "toc-entry"]
        if li.find("a"):
            li.find("a")["class"] = [*li.find("a").get("class", []), "nav-link"]
    if kind != "html":
        return soup
    h1_headers = soup.select(".toc-h1")
    if len(h1_headers) == 1:
        title = h1_headers[0]
        return "" if not title.select(".toc-h2") else title.find("ul")
    return soup


def _toc(items):
    return "<ul>\n" + "".join(items) + "</ul>\n"


def _toc_item(href, title, children=""):
    return (
        f'<li><a class="reference internal" href="{href}">{title}</a>{children}</li>\n'
    )


PAGE_TOCS = {
    # a single h1 (the page title) with sub-headers: only the sub-list is shown
    "title_with_subs": _toc(
        [
            _toc_item(
                "#",
                "The title",
                _toc(
                    [
                        _toc_item("#one", "One", _toc([_toc_item("#one-a", "One A")])),
                        _toc_item("#two", "Two"),
                    ]
                ),
            )
        ]
    ),
    # a single h1 with no sub-headers: no TOC at all
    "title_only": _toc([_toc_item("#", "The title")]),
    # several h1s: they are treated as sections
    "many_h1": _toc([_toc_item("#a", "A"), _toc_item("#b", "B")]),
    "empty": "",
    "whitespace": "\n",
}


@pytest.mark.parametrize("name", sorted(PAGE_TOCS))
@pytest.mark.parametrize("show_toc_level", [1, 2, 3])
@pytest.mark.parametrize("kind", ["html", "raw"])
def test_page_toc_matches_beautifulsoup(name, show_toc_level, kind) -> None:
    """The in-page TOC rewriter must match the old BeautifulSoup pipeline."""
    html = PAGE_TOCS[name]
    expected = bs4_page_toc_reference(html, show_toc_level, kind)
    root = rewrite_page_toc(html, show_toc_level=show_toc_level)
    got = render_page_toc(root) if kind == "html" else render_toctree(root)
    assert got == str(expected)
    # the template only renders the TOC when it is non-empty; that decision must
    # not change either
    assert (len(got) >= 1) == (len(expected) >= 1)
