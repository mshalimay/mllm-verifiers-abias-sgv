import re

_ID_RE = re.compile(r"\[(\d+)\]")
_QUOTED_RE = re.compile(r"'(.*?)'")


class AXTreeParser:
    def __init__(self):
        self.parsed_axtree = None

    def _parse_axtree(self, axtree_str):
        """
        Parse an 'axtree' string into an ordered list of nodes with indentation depth,
        plus quick maps for lookup and parent relations.
        Each node:
        {
            "id": int,
            "depth": int,     # number of leading tabs
            "role": str,      # e.g., "link", "textbox", "option", "StaticText", "RootWebArea", "Section", ...
            "name": str,      # the single-quoted label if present ('' allowed), else ''
            "attrs": str,     # the trailing comma-separated attributes portion (raw)
            "line": str,      # original line (trimmed)
        }
        """
        nodes = []
        lines = axtree_str.splitlines()

        for raw in lines:
            # count leading tabs = depth
            depth = len(raw) - len(raw.lstrip("\t"))
            line = raw.strip()

            # extract id
            m_id = _ID_RE.search(line)
            if not m_id:
                # Some top-level lines may be "RootWebArea '...' ..."
                # Synthesize an id of -1 so we can still keep hierarchy.
                # But we still want to try to parse role/name/attrs.
                node_id = -1
            else:
                node_id = int(m_id.group(1))

            # Grab the part after the closing bracket if present, otherwise whole line
            after = line
            if "]" in line:
                after = line.split("]", 1)[1].strip()

            # Role = first token before space or until a quote; handle "RootWebArea" specially
            # The 'after' string typically starts like: "link 'Text', clickable, visible"
            # or "RootWebArea 'Page', focused"
            # or "StaticText 'xxx'"
            # or "contentinfo ''"
            role = after.split(" ", 1)[0] if after else ""

            # Name inside single quotes (may be empty '')
            m_q = _QUOTED_RE.search(after)
            name = m_q.group(1) if m_q else ""

            # Attributes = everything after the first comma following the name,
            # else, everything after the name token if there's no comma
            attrs = ""
            if m_q:
                tail = after[m_q.end() :].lstrip()
                if tail.startswith(","):
                    attrs = tail[1:].strip()
                else:
                    # No comma after name; maybe nothing or flags right away
                    attrs = tail.strip()
            else:
                # Fallback: try to find first comma
                if "," in after:
                    attrs = after.split(",", 1)[1].strip()

            nodes.append(
                {
                    "id": node_id,
                    "depth": depth,
                    "role": role,
                    "name": name,
                    "attrs": attrs,
                    "line": line,
                }
            )

        # Build parent pointers by stack of depths
        parent = {}
        stack = []  # list of (depth, id)
        for n in nodes:
            d, nid = n["depth"], n["id"]
            while stack and stack[-1][0] >= d:
                stack.pop()
            # parent is the last item on stack (if any)
            if nid != -1:  # Only assign parents for real IDs
                parent[nid] = stack[-1][1] if stack else None
            stack.append((d, nid))

        # Quick lookup
        by_id = {n["id"]: n for n in nodes if n["id"] != -1}

        return nodes, by_id, parent

    def _attrs_to_flags(self, attrs_str):
        """
        Pull common boolean/kv flags from the raw attributes string.
        """
        text = attrs_str or ""
        flags = {
            "clickable": ("clickable" in text),
            "visible": ("visible" in text),
            "focused": ("focused" in text),
            "selected": None,
            "expanded": None,
            "hasPopup": None,
            "value": None,
        }
        m_sel = re.search(r"selected\s*=\s*(True|False)", text, re.IGNORECASE)
        if m_sel:
            flags["selected"] = m_sel.group(1).lower() == "true"
        m_exp = re.search(r"expanded\s*=\s*(True|False)", text, re.IGNORECASE)
        if m_exp:
            flags["expanded"] = m_exp.group(1).lower() == "true"
        m_popup = re.search(
            r"hasPopup\s*=\s*'?(True|False|menu|dialog|listbox|tree|grid|carousel)'?", text, re.IGNORECASE
        )
        if m_popup:
            flags["hasPopup"] = m_popup.group(1)
        m_val = re.search(r"value\s*=\s*'([^']*)'", text)
        if m_val:
            flags["value"] = m_val.group(1)
        return flags

    def _build_ancestors(self, by_id, parent_map, target_id):
        """
        Return a list of ancestor nodes from root->...->parent, each as {id, role, name}.
        """
        chain = []
        cur = parent_map.get(target_id)
        seen = set()
        while cur is not None and cur not in seen:
            seen.add(cur)
            n = by_id.get(cur)
            if not n:
                break
            chain.append({"id": cur, "role": n["role"], "name": n["name"]})
            cur = parent_map.get(cur)
        chain.reverse()
        return chain

    def get_semantic_info(self, step, bbox_id):
        """
        Derive semantic info for the given bbox_id using the step's axtree.
        Enriches action/reasoning with the node's role/name/flags and its ancestors path.
        """
        semantic_info = {}

        axtree = step.get("axtree", "")
        if not axtree:
            semantic_info["node"] = None
            semantic_info["ancestors"] = []
            semantic_info["summary"] = None
            return semantic_info

        # Parse the axtree and locate the node for this bbox_id
        try:
            target_id = int(bbox_id)
        except Exception:
            # bbox_id could already be an int-like string, but be defensive
            try:
                target_id = int(str(bbox_id))
            except Exception:
                target_id = None

        nodes, by_id, parent_map = self._parse_axtree(axtree)

        node = by_id.get(target_id)
        if not node:
            # Could not match; still return basics.
            semantic_info["node"] = None
            semantic_info["ancestors"] = []
            semantic_info["summary"] = None
            return semantic_info

        flags = self._attrs_to_flags(node.get("attrs", ""))

        # Build ancestors (breadcrumb)
        ancestors = self._build_ancestors(by_id, parent_map, target_id)

        # Human-readable summary (role, name, key flags/values)
        parts = [node["role"]]
        if node["name"] != "":
            parts.append(f"'{node['name']}'")
        # attach salient flags/attrs
        if flags.get("value") is not None:
            parts.append(f"value='{flags['value']}'")
        if flags.get("hasPopup") is not None:
            parts.append(f"hasPopup={flags['hasPopup']}")
        if flags.get("expanded") is not None:
            parts.append(f"expanded={flags['expanded']}")
        if flags.get("selected") is not None:
            parts.append(f"selected={flags['selected']}")
        if flags.get("clickable"):
            parts.append("clickable")
        if flags.get("visible"):
            parts.append("visible")

        summary = " ".join(parts)

        semantic_info["node"] = {
            "id": target_id,
            "role": node["role"],
            "name": node["name"],
            "raw_attrs": node["attrs"],
            "flags": flags,
            "line": node["line"],
        }
        semantic_info["ancestors"] = ancestors
        semantic_info["summary"] = summary
        return semantic_info
