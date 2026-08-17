"""BeautifulSoup-free rewriting of the sidebar toctree HTML.

Sphinx hands us the resolved sidebar toctree as an HTML fragment (from
``builder.render_partial``). The theme then reshapes it: pairing ``current``
with ``active``, dropping in-page-anchor entries, adding Bootstrap classes and
``<details>``/``<summary>`` disclosure widgets, ... On a big site (thousands of
sidebar entries written onto thousands of pages) doing that with BeautifulSoup
is the dominant cost of the HTML write phase, mostly because the finished tree
has to be re-serialized for every single page.

This module does the same work on top of :mod:`html.parser` (which is exactly
the parser BeautifulSoup's ``"html.parser"`` backend wraps) and, crucially, can
emit a :class:`SidebarTemplate`: the finished HTML as one string plus a small
table of "slots" -- the handful of substrings that have to change to move the
"current page" markers from one entry to another. Rendering the sidebar for
another page in the same directory is then a few string slices instead of a
tree walk plus a full re-serialization.

The serialization intentionally mimics BeautifulSoup's default (``"minimal"``)
output formatter byte for byte: attributes sorted by name, ``&``/``<``/``>``
escaped, attribute values double-quoted unless they contain a double quote, and
void elements written as ``<br/>``.
"""

import re

from html.parser import HTMLParser


# Tags BeautifulSoup's HTMLTreeBuilder treats as empty-element tags; they are
# never pushed on the element stack and are serialized as ``<br/>``.
_VOID_ELEMENTS = frozenset(
    {
        "area",
        "base",
        "basefont",
        "bgsound",
        "br",
        "col",
        "command",
        "embed",
        "frame",
        "hr",
        "image",
        "img",
        "input",
        "isindex",
        "keygen",
        "link",
        "menuitem",
        "meta",
        "nextid",
        "param",
        "source",
        "spacer",
        "track",
        "wbr",
    }
)

# Attributes BeautifulSoup stores (and re-emits) as whitespace-separated lists.
_LIST_ATTRIBUTES = frozenset({"accesskey", "class", "dropzone"})
_LIST_ATTRIBUTES_BY_TAG = {
    "a": frozenset({"rel", "rev"}),
    "area": frozenset({"rel"}),
    "form": frozenset({"accept-charset"}),
    "icon": frozenset({"sizes"}),
    "iframe": frozenset({"sandbox"}),
    "link": frozenset({"rel", "rev"}),
    "object": frozenset({"archive"}),
    "output": frozenset({"for"}),
    "td": frozenset({"headers"}),
    "th": frozenset({"headers"}),
}

# elements whose markup depends on which page the sidebar is rendered for
_MARKED_ELEMENTS = frozenset({"a", "details", "li", "ul"})

_NONWHITESPACE = re.compile(r"\S+")
_ESCAPES = {"&": "&amp;", "<": "&lt;", ">": "&gt;"}
_ESCAPE_RE = re.compile("[&<>]")


def _escape(text: str) -> str:
    """Escape a string the way BeautifulSoup's "minimal" formatter does."""
    return _ESCAPE_RE.sub(lambda match: _ESCAPES[match.group()], text)


def _quote_attr(value: str) -> str:
    """Escape *and* quote an attribute value the way BeautifulSoup does."""
    value = _escape(value)
    if '"' in value:
        if "'" in value:
            return '"' + value.replace('"', "&quot;") + '"'
        return "'" + value + "'"
    return '"' + value + '"'


class _RawText(str):
    """Markup (a comment, a declaration) that is emitted verbatim."""

    __slots__ = ()


class _Element:
    """A minimal HTML element: name, attributes, children, parent."""

    __slots__ = ("attrs", "children", "name", "parent")

    def __init__(self, name: str, attrs: dict, parent=None):
        self.name = name
        self.attrs = attrs
        self.children = []
        self.parent = parent

    @property
    def classes(self) -> list:
        """Return the (possibly empty) list of CSS classes."""
        return self.attrs.get("class", [])

    def start_tag(self, attrs: dict | None = None) -> str:
        """Serialize the opening tag, optionally with substitute attributes."""
        if attrs is None:
            attrs = self.attrs
        out = ["<", self.name]
        for key in sorted(attrs):
            value = attrs[key]
            if value is None:
                out.append(" " + key)
                continue
            if isinstance(value, list):
                value = " ".join(value)
            out.append(" " + key + "=" + _quote_attr(value))
        if self.name in _VOID_ELEMENTS:
            out.append("/")
        out.append(">")
        return "".join(out)

    def iter_children(self):
        """Yield the child elements (skipping text nodes)."""
        for child in self.children:
            if not isinstance(child, str):
                yield child

    def find_child(self, name: str):
        """Return the first direct child element named ``name``, or None."""
        for child in self.children:
            if not isinstance(child, str) and child.name == name:
                return child
        return None

    def find(self, name: str):
        """Return the first descendant element named ``name``, or None."""
        for child in self.children:
            if isinstance(child, str):
                continue
            if child.name == name:
                return child
            found = child.find(name)
            if found is not None:
                return found
        return None

    def iter_descendants(self, name: str):
        """Yield descendant elements named ``name`` in document order."""
        for child in self.children:
            if isinstance(child, str):
                continue
            if child.name == name:
                yield child
            yield from child.iter_descendants(name)

    def iter_all(self):
        """Yield all descendant elements in document order."""
        for child in self.children:
            if isinstance(child, str):
                continue
            yield child
            yield from child.iter_all()

    def adopt(self, children: list) -> None:
        """Take over ``children``, re-parenting the elements among them."""
        self.children = children
        for child in children:
            if not isinstance(child, str):
                child.parent = self


class _TreeParser(HTMLParser):
    """Build an :class:`_Element` tree, mimicking BeautifulSoup's html.parser."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = _Element("[document]", {})
        self._stack = [self.root]

    def handle_starttag(self, tag: str, attrs: list) -> None:
        """Open an element (void elements are not pushed on the stack)."""
        as_list = _LIST_ATTRIBUTES_BY_TAG.get(tag, frozenset())
        parsed = {}
        for key, value in attrs:
            value = value or ""
            if key in _LIST_ATTRIBUTES or key in as_list:
                value = _NONWHITESPACE.findall(value)
            parsed[key] = value  # a repeated attribute keeps its last value
        element = _Element(tag, parsed, self._stack[-1])
        self._stack[-1].children.append(element)
        if tag not in _VOID_ELEMENTS:
            self._stack.append(element)

    def handle_endtag(self, tag: str) -> None:
        """Close the most recently opened element with this name, if any."""
        stack = self._stack
        for index in range(len(stack) - 1, 0, -1):
            if stack[index].name == tag:
                del stack[index:]
                return

    def handle_data(self, data: str) -> None:
        """Append a text node."""
        self._stack[-1].children.append(data)

    def handle_comment(self, data: str) -> None:
        """Append a comment node."""
        self._stack[-1].children.append(_RawText(f"<!--{data}-->"))

    def handle_decl(self, decl: str) -> None:
        """Append a declaration node."""
        self._stack[-1].children.append(_RawText(f"<!{decl}>"))


def _text_out(node: str) -> str:
    """Return the serialization of a text node."""
    return node if type(node) is _RawText else _escape(node)


def _serialize(element: _Element, out: list) -> None:
    """Append the serialization of ``element``'s children to ``out``."""
    for child in element.children:
        if isinstance(child, str):
            out.append(_text_out(child))
            continue
        out.append(child.start_tag())
        if child.name not in _VOID_ELEMENTS:
            _serialize(child, out)
            out.append(f"</{child.name}>")


# -- the transforms ----------------------------------------------------------


def _pair_current_with_active(root: _Element) -> None:
    """Pair "current" with "active" since that's what we use w/ bootstrap."""
    for item in root.iter_descendants("li"):
        classes = item.classes
        if "current" in classes:
            item.attrs["class"] = [*classes, "active"]


def _drop_in_page_entries(element: _Element) -> None:
    """Remove sidebar links to sub-headers on the page.

    An ``<li>`` goes away (with its whole subtree) when the first ``<a>`` below
    it points at an in-page anchor, i.e. its href contains a ``#`` but is not
    just ``"#"`` (which is how the current page's own entry is rendered).
    """
    kept = []
    dropped = False
    for child in element.children:
        if not isinstance(child, str) and child.name == "li":
            anchor = child.find("a")
            if anchor is not None:
                href = anchor.attrs.get("href", "")
                if "#" in href and href != "#":
                    dropped = True
                    continue
        kept.append(child)
        if not isinstance(child, str):
            _drop_in_page_entries(child)
    if dropped:
        element.children = kept


def _add_sidenav_classes(root: _Element) -> None:
    """Add bootstrap classes to the top-level ``<ul>`` elements."""
    for child in root.iter_children():
        if child.name == "ul":
            child.attrs["class"] = [*child.classes, "nav", "bd-sidenav"]


def _wrap_parts(root: _Element) -> None:
    """Wrap each part (caption + the list that follows it) in ``li.toctree-l0``.

    Only used with ``show_nav_level=0``, which makes parts collapsible. The
    result replaces the document root's children entirely (anything that is
    neither a caption nor the ``<ul>`` following one is dropped), matching the
    previous BeautifulSoup implementation.
    """
    captions = [
        element
        for element in root.iter_descendants("p")
        if "caption" in element.classes
    ]
    if not captions:
        return
    wrapper = _Element("ul", {"class": ["list-caption"]}, root)
    for caption in captions:
        siblings = caption.parent.children
        contents = [caption]
        for sibling in siblings[siblings.index(caption) + 1 :]:
            # Assume that the next <ul> element is the TOC list for this part
            if not isinstance(sibling, str) and sibling.name == "ul":
                contents.append(sibling)
                break
        item = _Element("li", {"class": ["toctree-l0"]}, wrapper)
        item.adopt(contents)
        wrapper.children.append(item)
    root.children = [wrapper]


def _add_collapse_widgets(root: _Element) -> None:
    """Add ``<details>``/``<summary>`` disclosure widgets to nested entries.

    Mirrors :func:`pydata_sphinx_theme.toctree.add_collapse_checkboxes`; see
    that function for the (annotated) shape of the resulting markup.
    """
    for element in list(root.iter_descendants("li")):
        classes = element.classes
        is_current = "current" in classes

        # expanding the parent part explicitly, if present
        if is_current:
            parent = element.parent
            while parent is not None:
                if parent.name == "li" and "toctree-l0" in parent.classes:
                    parent.find("details").attrs["open"] = None
                    break
                parent = parent.parent

        # Nothing more to do, unless this has "children"
        if element.find("ul") is None:
            continue

        # Add a class to indicate that this has children.
        element.attrs["class"] = [*classes, "has-children"]

        # Create <details> and put the entire subtree into it
        details = _Element("details", {}, element)
        details.adopt(element.children)
        element.children = [details]

        # Hoist the link to the top if there is one
        for child in details.iter_children():
            if child.name == "a" and "reference" in child.classes:
                details.children.remove(child)
                child.parent = element
                element.children.insert(0, child)
                break

        # Create <summary> with chevron icon
        summary = _Element("summary", {}, details)
        span = _Element(
            "span",
            # This element and the chevron it contains are purely decorative;
            # the actual expand/collapse functionality is delegated to the
            # <summary> tag
            {"class": ["toctree-toggle"], "role": "presentation"},
            summary,
        )
        span.children.append(
            _Element("i", {"class": ["fa-solid", "fa-chevron-down"]}, span)
        )
        summary.children.append(span)

        # Prepend the section heading (if there is one) to <summary>, so that
        # the heading text and the chevron are both clickable
        for child in details.iter_children():
            if child.name == "p" and "caption" in child.classes:
                details.children.remove(child)
                child.parent = summary
                summary.children.insert(0, child)
                break

        details.children.insert(0, summary)

        # If this TOC node has a "current" class, be expanded by default
        if is_current:
            details.attrs["open"] = "open"


def _open_to_nav_level(root: _Element, show_nav_level: int) -> None:
    """Open the sidebar navigation to the proper depth."""
    if show_nav_level < 1:
        return
    levels = {f"toctree-l{level}" for level in range(show_nav_level)}
    for item in root.iter_descendants("li"):
        if levels.intersection(item.classes):
            details = item.find_child("details")
            if details is not None:
                details.attrs["open"] = "open"


def rewrite_toctree(html: str, *, kind: str, show_nav_level: int) -> _Element:
    """Parse and rewrite a Sphinx toctree fragment; return the document root."""
    parser = _TreeParser()
    parser.feed(html)
    parser.close()
    root = parser.root
    _pair_current_with_active(root)
    _drop_in_page_entries(root)
    if kind == "sidebar":
        _add_sidenav_classes(root)
        if show_nav_level == 0:
            _wrap_parts(root)
        _add_collapse_widgets(root)
        _open_to_nav_level(root, show_nav_level)
    return root


def render_toctree(root: _Element) -> str:
    """Serialize a rewritten toctree the way BeautifulSoup's ``str()`` would."""
    out = []
    _serialize(root, out)
    return "".join(out)


def render_element(element: _Element) -> str:
    """Serialize a single element (tag included) and its subtree."""
    out = [element.start_tag()]
    if element.name not in _VOID_ELEMENTS:
        _serialize(element, out)
        out.append(f"</{element.name}>")
    return "".join(out)


# -- the in-page ("On this page") table of contents ---------------------------


def _add_header_levels(ul: _Element | None, level: int, show_toc_level: int) -> None:
    """Add ``toc-hN`` (and visibility) classes to a nested ``<ul>``."""
    if ul is None:
        return
    if level <= show_toc_level + 1:
        ul.attrs["class"] = [*ul.classes, "pst-show_toc_level"]
    for item in ul.iter_children():
        if item.name != "li":
            continue
        item.attrs["class"] = [*item.classes, f"toc-h{level}"]
        _add_header_levels(item.find_child("ul"), level + 1, show_toc_level)


def rewrite_page_toc(html: str, *, show_toc_level: int) -> _Element:
    """Parse and rewrite Sphinx's in-page TOC; return the document root."""
    parser = _TreeParser()
    parser.feed(html)
    parser.close()
    root = parser.root
    _add_header_levels(root.find("ul"), 1, show_toc_level)
    # Add in CSS classes for bootstrap
    for ul in root.iter_descendants("ul"):
        ul.attrs["class"] = [*ul.classes, "nav", "section-nav", "flex-column"]
    for item in root.iter_descendants("li"):
        item.attrs["class"] = [*item.classes, "nav-item", "toc-entry"]
        anchor = item.find("a")
        if anchor is not None:
            anchor.attrs["class"] = [*anchor.classes, "nav-link"]
    return root


def render_page_toc(root: _Element) -> str:
    """Serialize a rewritten in-page TOC, dropping a lone page title.

    If the page has a single ``h1`` we assume it is the page title: its own
    sub-list is returned (and nothing at all if it has no sub-headers, since
    then there is no TOC worth showing).
    """
    titles = [element for element in root.iter_all() if "toc-h1" in element.classes]
    if len(titles) != 1:
        return render_toctree(root)
    title = titles[0]
    if not any("toc-h2" in element.classes for element in title.iter_all()):
        return ""
    sublist = title.find("ul")
    return "" if sublist is None else render_element(sublist)


# -- "current page" markers --------------------------------------------------


def _statically_open(item: _Element, show_nav_level: int) -> bool:
    """Whether this ``<li>``'s disclosure widget is open regardless of page."""
    return any(f"toctree-l{level}" in item.classes for level in range(show_nav_level))


def _current_chain(anchor: _Element, show_nav_level: int) -> list:
    """Return the elements that carry "current page" markers for ``anchor``.

    That is the anchor itself plus, walking up to the root, every ``<li>`` and
    ``<ul>`` above it (each of which gains a ``current`` class, and ``<li>``
    also ``active``) and every ``<details>`` on the chain (which is opened).
    Note that the current entry's own ``<details>`` is a *sibling* of its
    anchor, not an ancestor. Two kinds of element are deliberately skipped,
    matching a fresh build: the ``li.toctree-l0`` / ``ul.list-caption``
    wrappers this theme synthesizes around parts are never marked current (only
    their ``<details>`` is opened), and disclosure widgets that
    ``show_nav_level`` keeps open on every page never move.
    """
    chain = [anchor]
    node = anchor.parent
    while node is not None and node.name != "[document]":
        if node.name == "li":
            if "toctree-l0" not in node.classes:
                chain.append(node)
            details = node.find_child("details")
            if details is not None and not _statically_open(node, show_nav_level):
                chain.append(details)
        elif node.name == "ul" and "list-caption" not in node.classes:
            chain.append(node)
        node = node.parent
    return chain


def _markers_off(element: _Element, show_nav_level: int) -> dict | None:
    """Return ``element``'s attributes with any "current page" markers removed.

    Returns None when the element carries no such markers, so that the caller
    can serialize it as-is.
    """
    attrs = element.attrs
    if element.name == "details":
        if "open" not in attrs or _statically_open(element.parent, show_nav_level):
            return None
        attrs = dict(attrs)
        del attrs["open"]
        return attrs
    classes = attrs.get("class", ())
    if element.name == "li":
        stripped = [name for name in classes if name not in ("current", "active")]
    else:  # "a" and "ul"
        stripped = [name for name in classes if name != "current"]
    if len(stripped) == len(classes):
        return None
    attrs = dict(attrs)
    if stripped:
        attrs["class"] = stripped
    else:
        del attrs["class"]
    return attrs


def _markers_on(element: _Element, attrs: dict, is_part_details: bool) -> dict:
    """Return ``attrs`` with this element's "current page" markers added."""
    attrs = dict(attrs)
    if element.name == "details":
        # a fresh build gives an open part a bare `open` attribute
        attrs["open"] = None if is_part_details else "open"
        return attrs
    classes = attrs.get("class", [])
    if element.name == "li":
        # match the class order of a freshly built toctree, where Sphinx adds
        # "current" right after "toctree-l*"
        attrs["class"] = [*classes[:1], "current", "active", *classes[1:]]
    else:  # "a" and "ul"
        attrs["class"] = ["current", *classes]
        if element.name == "a":
            attrs["href"] = "#"  # the current page's entry links to itself
    return attrs


class SidebarTemplate:
    """A rendered sidebar plus the slots needed to re-target it to other pages.

    ``html`` is the finished sidebar for one page of a directory, with every
    "current page" marker removed. ``slots[i]`` is ``(start, end, replacement)``
    -- replacing ``html[start:end]`` with ``replacement`` re-adds one marker --
    and ``by_href`` maps the href of each entry to the slots that mark that
    entry (and its ancestors) as the current page. ``self_anchors`` re-points
    the entries of the page ``html`` was built for, whose relative href depends
    on the page being rendered.
    """

    __slots__ = ("by_href", "html", "self_anchors", "slots")

    def __init__(self, html: str, slots: list, by_href: dict, self_anchors: list):
        self.html = html
        self.slots = slots
        self.by_href = by_href
        self.self_anchors = self_anchors

    def render(self, href_new: str, href_old: str) -> str | None:
        """Return the sidebar with the current-page markers moved to ``href_new``.

        ``href_old`` is where the page this template was built for lives, as
        seen from the page being rendered. Returns None if there is no entry for
        ``href_new`` (e.g. it was pruned by ``maxdepth``), in which case the
        caller has to build the sidebar the slow way.
        """
        indices = self.by_href.get(href_new)
        if indices is None:
            return None
        edits = {}
        for start, end, prefix, suffix in self.self_anchors:
            edits[start] = (end, prefix + _quote_attr(href_old) + suffix)
        slots = self.slots
        for index in indices:
            start, end, replacement = slots[index]
            edits[start] = (end, replacement)
        html = self.html
        out = []
        position = 0
        for start in sorted(edits):
            end, replacement = edits[start]
            out.append(html[position:start])
            out.append(replacement)
            position = end
        out.append(html[position:])
        return "".join(out)


class _TemplateBuilder:
    """Serialize a rewritten toctree while collecting its current-page slots."""

    def __init__(self, show_nav_level: int, self_href: str):
        self.show_nav_level = show_nav_level
        self.self_href = self_href
        self.out = []
        self.position = 0
        self.slots = []
        self.slot_of = {}
        self.self_anchors = []

    def emit(self, text: str) -> None:
        """Append literal text to the output."""
        self.out.append(text)
        self.position += len(text)

    def build(self, root: _Element) -> SidebarTemplate:
        """Serialize ``root`` markers-off and return the finished template."""
        entries = []
        slotted = set()
        for anchor in root.iter_descendants("a"):
            if "external" in anchor.classes:
                continue
            href = anchor.attrs.get("href")
            if href is None or ("#" in href and href != "#"):
                continue
            chain = _current_chain(anchor, self.show_nav_level)
            entries.append((href, chain))
            slotted.update(id(element) for element in chain)
        self._walk(root, slotted)
        by_href = {}
        slot_of = self.slot_of
        for href, chain in entries:
            indices = by_href.setdefault(self.self_href if href == "#" else href, [])
            indices.extend(slot_of[id(element)] for element in chain)
        for href, indices in by_href.items():
            by_href[href] = sorted(set(indices))
        if self.self_href in by_href:
            # `get_relative_uri(page, page)` is "", not the page's own basename
            by_href[""] = by_href[self.self_href]
        return SidebarTemplate(
            "".join(self.out), self.slots, by_href, self.self_anchors
        )

    def _walk(self, element: _Element, slotted: set) -> None:
        for child in element.children:
            if isinstance(child, str):
                self.emit(_text_out(child))
                continue
            if child.name in _MARKED_ELEMENTS:
                attrs = _markers_off(child, self.show_nav_level)
                if id(child) in slotted:
                    self._emit_slot(child, child.attrs if attrs is None else attrs)
                else:
                    self.emit(child.start_tag(attrs))
            else:
                self.emit(child.start_tag())
            if child.name not in _VOID_ELEMENTS:
                self._walk(child, slotted)
                self.emit(f"</{child.name}>")

    def _emit_slot(self, element: _Element, attrs: dict) -> None:
        is_self_anchor = element.name == "a" and attrs["href"] == "#"
        if is_self_anchor:
            attrs = dict(attrs)
            attrs["href"] = self.self_href
        is_part_details = element.name == "details" and (
            "toctree-l0" in element.parent.classes
        )
        off_tag = element.start_tag(attrs)
        on_tag = element.start_tag(_markers_on(element, attrs, is_part_details))
        start = self.position
        self.emit(off_tag)
        self.slot_of[id(element)] = len(self.slots)
        self.slots.append((start, self.position, on_tag))
        if is_self_anchor:
            # this href has to be recomputed per page for builders whose page
            # URIs are not flat (e.g. the "dirhtml" builder)
            quoted = _quote_attr(self.self_href)
            cut = off_tag.index("href=" + quoted) + len("href=")
            self.self_anchors.append(
                (start, self.position, off_tag[:cut], off_tag[cut + len(quoted) :])
            )


def build_template(
    root: _Element, *, show_nav_level: int, self_href: str
) -> SidebarTemplate:
    """Build a re-targetable :class:`SidebarTemplate` from a rewritten toctree."""
    return _TemplateBuilder(show_nav_level, self_href).build(root)
