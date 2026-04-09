"""Rust source code loader using tree-sitter parsing."""

import logging
from typing import NamedTuple

import tree_sitter_rust
from tree_sitter import Language, Parser

from qdrant_indexer.models import CodeSymbol

from .tree_sitter_base import TreeSitterCodeLoader

logger = logging.getLogger(__name__)


class _CommonFields(NamedTuple):
    """Shared fields extracted from every Rust AST node."""

    name: str
    source: str
    docstring: str | None
    visibility: str
    attributes: list[str]
    generics: list[str]
    lifetimes: list[str]
    where_clause: str | None


class RustCodeLoader(TreeSitterCodeLoader):
    """Loader for Rust source files using tree-sitter.

    Extracts functions, structs, enums, traits, impl blocks, methods,
    type aliases, constants, statics, and macros from Rust source code.
    Also extracts doc comments (/// and /** */).
    """

    def __init__(self) -> None:
        """Initialize the Rust parser with tree-sitter."""
        self._rust_lang = Language(tree_sitter_rust.language())
        self._parser = Parser(self._rust_lang)

    def _walk_node(
        self,
        node,
        content_bytes: bytes,
        symbols: list[CodeSymbol],
        parent_name: str | None,
    ) -> None:
        """Recursively walk AST nodes to extract symbols.

        Args:
            node: Tree-sitter node to process.
            content_bytes: Source code as bytes.
            symbols: List to append extracted symbols to.
            parent_name: Name of parent impl/trait if inside one.
        """
        if node.type == "function_item":
            symbols.append(
                self._extract_function(node, content_bytes, parent_name)
            )

        elif node.type == "struct_item":
            symbols.append(self._extract_struct(node, content_bytes))

        elif node.type == "enum_item":
            symbols.append(self._extract_enum(node, content_bytes))

        elif node.type == "trait_item":
            trait_symbol = self._extract_trait(node, content_bytes)
            symbols.append(trait_symbol)
            # Process methods inside trait
            self._extract_trait_methods(node, content_bytes, symbols, trait_symbol.name)

        elif node.type == "impl_item":
            impl_symbol = self._extract_impl(node, content_bytes)
            symbols.append(impl_symbol)
            # Process methods inside impl
            self._extract_impl_methods(node, content_bytes, symbols, impl_symbol.name)

        elif node.type == "type_item":
            symbols.append(self._extract_type_alias(node, content_bytes))

        elif node.type == "const_item":
            symbols.append(self._extract_const(node, content_bytes))

        elif node.type == "static_item":
            symbols.append(self._extract_static(node, content_bytes))

        elif node.type == "macro_definition":
            symbols.append(self._extract_macro(node, content_bytes))

        elif node.type == "mod_item":
            # Handle inline modules
            mod_symbol = self._extract_module(node, content_bytes)
            if mod_symbol:
                symbols.append(mod_symbol)
                # Process items inside inline module
                body = node.child_by_field_name("body")
                if body:
                    for child in body.children:
                        self._walk_node(child, content_bytes, symbols, mod_symbol.name)
                    return  # Don't recurse again below

        else:
            # Recurse into children for other node types
            for child in node.children:
                self._walk_node(child, content_bytes, symbols, parent_name)

    def _extract_doc_comment(self, node, content_bytes: bytes) -> str | None:
        """Extract doc comment above a node.

        Handles both /// line comments and /** */ block comments.

        Args:
            node: The node to find doc comment for.
            content_bytes: Source code as bytes.

        Returns:
            Cleaned doc comment content or None if not found.
        """
        doc_lines = []
        prev = node.prev_sibling

        while prev:
            if prev.type == "line_comment":
                comment_text = self._get_node_text(prev, content_bytes)
                if comment_text.startswith("///"):
                    # Doc comment - add to front (we're going backwards)
                    doc_lines.insert(0, comment_text[3:].strip())
                elif comment_text.startswith("//!"):
                    # Inner doc comment - also capture
                    doc_lines.insert(0, comment_text[3:].strip())
                else:
                    # Regular comment, stop here
                    break
            elif prev.type == "block_comment":
                comment_text = self._get_node_text(prev, content_bytes)
                if comment_text.startswith("/**") and not comment_text.startswith("/**/"):
                    return self._clean_block_comment(comment_text)
                elif comment_text.startswith("/*!"):
                    return self._clean_block_comment(comment_text)
                else:
                    break
            elif prev.type == "attribute_item" or prev.type == "inner_attribute_item":
                # Skip attributes, they can be between doc and item
                prev = prev.prev_sibling
                continue
            else:
                break
            prev = prev.prev_sibling

        return "\n".join(doc_lines) if doc_lines else None

    def _extract_visibility(self, node, content_bytes: bytes) -> str:
        """Extract visibility modifier from a node.

        Args:
            node: Node that may have visibility modifier.
            content_bytes: Source code as bytes.

        Returns:
            Visibility string: 'pub', 'pub(crate)', 'pub(super)', etc., or 'private'.
        """
        for child in node.children:
            if child.type == "visibility_modifier":
                vis_text = self._get_node_text(child, content_bytes)
                return vis_text
        return "private"

    def _extract_attributes(self, node, content_bytes: bytes) -> list[str]:
        """Extract attributes from preceding siblings.

        Args:
            node: Node to find attributes for.
            content_bytes: Source code as bytes.

        Returns:
            List of attribute strings (e.g., ['#[derive(Debug)]', '#[cfg(test)]']).
        """
        attributes = []
        prev = node.prev_sibling

        while prev:
            if prev.type == "attribute_item":
                attr_text = self._get_node_text(prev, content_bytes)
                attributes.insert(0, attr_text)
            elif prev.type == "line_comment" or prev.type == "block_comment":
                # Skip comments between attributes
                prev = prev.prev_sibling
                continue
            else:
                break
            prev = prev.prev_sibling

        return attributes

    def _parse_derives(self, attributes: list[str]) -> list[str]:
        """Parse derive macros from attributes.

        Args:
            attributes: List of attribute strings.

        Returns:
            List of derived trait names.
        """
        derives = []
        for attr in attributes:
            if "derive(" in attr:
                # Extract content between derive( and )
                start = attr.find("derive(") + 7
                end = attr.find(")", start)
                if end > start:
                    derive_content = attr[start:end]
                    # Split by comma and clean up
                    for item in derive_content.split(","):
                        item = item.strip()
                        if item:
                            derives.append(item)
        return derives

    def _extract_generics(self, node, content_bytes: bytes) -> tuple[list[str], list[str], str | None]:
        """Extract generic parameters, lifetimes, and where clause.

        Args:
            node: Node that may have generics.
            content_bytes: Source code as bytes.

        Returns:
            Tuple of (generics, lifetimes, where_clause).
        """
        generics = []
        lifetimes = []
        where_clause = None

        type_params = node.child_by_field_name("type_parameters")
        if type_params:
            for child in type_params.children:
                if child.type == "type_parameter":
                    # type_parameter contains type_identifier and optional trait_bounds
                    generics.append(self._get_node_text(child, content_bytes))
                elif child.type == "constrained_type_parameter":
                    generics.append(self._get_node_text(child, content_bytes))
                elif child.type == "lifetime_parameter":
                    # lifetime_parameter contains a lifetime node
                    lifetimes.append(self._get_node_text(child, content_bytes))
                elif child.type == "lifetime":
                    # Direct lifetime (less common)
                    lifetimes.append(self._get_node_text(child, content_bytes))

        # Find where clause
        for child in node.children:
            if child.type == "where_clause":
                where_clause = self._get_node_text(child, content_bytes)
                # Clean up the where clause
                if where_clause.startswith("where"):
                    where_clause = where_clause[5:].strip()
                break

        return generics, lifetimes, where_clause

    def _extract_common_fields(
        self,
        node,
        content_bytes: bytes,
        *,
        name_field: str = "name",
        name_default: str = "unknown",
        with_visibility: bool = True,
        with_generics: bool = True,
    ) -> _CommonFields:
        """Extract fields shared by every Rust symbol extractor.

        Centralises the five-line preamble that previously appeared in every
        ``_extract_*`` method: name lookup, source text, doc comment, visibility
        modifier, outer attributes, and (optionally) generic parameters.

        Args:
            node: Tree-sitter AST node for the item.
            content_bytes: Full source file encoded as UTF-8.
            name_field: The tree-sitter field name used to locate the identifier
                child, typically ``"name"``.
            name_default: Fallback when the name field is absent (``"unknown"``
                for most symbols, ``"UNKNOWN"`` for consts and statics).
            with_visibility: When False the returned visibility is ``"private"``
                without inspecting the node (used where the caller will override
                or where visibility is semantically meaningless).
            with_generics: When True, generic parameters, lifetime parameters,
                and the where clause are extracted; when False all three are
                returned as empty / None (saves work for const/static/macro/mod).

        Returns:
            A ``_CommonFields`` NamedTuple with all shared fields populated.
        """
        name_node = node.child_by_field_name(name_field)
        name = self._get_node_text(name_node, content_bytes) or name_default
        source = self._get_node_text(node, content_bytes)
        docstring = self._extract_doc_comment(node, content_bytes)
        visibility = self._extract_visibility(node, content_bytes) if with_visibility else "private"
        attributes = self._extract_attributes(node, content_bytes)

        if with_generics:
            generics, lifetimes, where_clause = self._extract_generics(node, content_bytes)
        else:
            generics, lifetimes, where_clause = [], [], None

        return _CommonFields(
            name=name,
            source=source,
            docstring=docstring,
            visibility=visibility,
            attributes=attributes,
            generics=generics,
            lifetimes=lifetimes,
            where_clause=where_clause,
        )

    def _extract_function(
        self, node, content_bytes: bytes, parent_name: str | None
    ) -> CodeSymbol:
        """Extract function symbol.

        Args:
            node: Function item node.
            content_bytes: Source code as bytes.
            parent_name: Parent impl/trait name if this is a method.

        Returns:
            CodeSymbol representing the function.
        """
        cf = self._extract_common_fields(node, content_bytes)
        name, source, docstring, visibility, attributes = (
            cf.name, cf.source, cf.docstring, cf.visibility, cf.attributes
        )
        generics, lifetimes, where_clause = cf.generics, cf.lifetimes, cf.where_clause

        # Check for async/unsafe/const in function_modifiers
        is_async = False
        is_unsafe = False
        is_const = False
        for child in node.children:
            if child.type == "function_modifiers":
                for modifier in child.children:
                    if modifier.type == "async":
                        is_async = True
                    elif modifier.type == "unsafe":
                        is_unsafe = True
                    elif modifier.type == "const":
                        is_const = True

        # Build signature
        signature = self._build_function_signature(node, content_bytes, visibility, is_async, is_unsafe, is_const)

        qualified_name = f"{parent_name}.{name}" if parent_name else name
        symbol_type = "method" if parent_name else "function"

        return CodeSymbol(
            name=name,
            qualified_name=qualified_name,
            symbol_type=symbol_type,
            content=source,
            language="rust",
            docstring=docstring,
            signature=signature,
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            parent=parent_name,
            visibility=visibility,
            metadata={
                "is_async": is_async,
                "is_unsafe": is_unsafe,
                "is_const": is_const,
                "generics": generics,
                "lifetimes": lifetimes,
                "where_clause": where_clause,
                "attributes": attributes,
            },
        )

    def _build_function_signature(
        self, node, content_bytes: bytes, visibility: str, is_async: bool, is_unsafe: bool, is_const: bool
    ) -> str:
        """Build function signature string.

        Args:
            node: Function item node.
            content_bytes: Source code as bytes.
            visibility: Visibility modifier.
            is_async: Whether function is async.
            is_unsafe: Whether function is unsafe.
            is_const: Whether function is const.

        Returns:
            Function signature string.
        """
        parts = []

        if visibility != "private":
            parts.append(visibility)
        if is_const:
            parts.append("const")
        if is_async:
            parts.append("async")
        if is_unsafe:
            parts.append("unsafe")
        parts.append("fn")

        name_node = node.child_by_field_name("name")
        name = self._get_node_text(name_node, content_bytes) or "unknown"
        parts.append(name)

        # Add type parameters
        type_params = node.child_by_field_name("type_parameters")
        if type_params:
            parts[-1] += self._get_node_text(type_params, content_bytes)

        # Add parameters
        params_node = node.child_by_field_name("parameters")
        params = self._get_node_text(params_node, content_bytes) or "()"
        parts[-1] += params

        # Add return type
        return_type = node.child_by_field_name("return_type")
        if return_type:
            parts.append("->")
            parts.append(self._get_node_text(return_type, content_bytes))

        return " ".join(parts)

    def _extract_struct(self, node, content_bytes: bytes) -> CodeSymbol:
        """Extract struct symbol.

        Args:
            node: Struct item node.
            content_bytes: Source code as bytes.

        Returns:
            CodeSymbol representing the struct.
        """
        cf = self._extract_common_fields(node, content_bytes)
        name, source, docstring, visibility, attributes = (
            cf.name, cf.source, cf.docstring, cf.visibility, cf.attributes
        )
        generics, lifetimes, where_clause = cf.generics, cf.lifetimes, cf.where_clause
        derives = self._parse_derives(attributes)

        # Build signature
        sig_parts = []
        if visibility != "private":
            sig_parts.append(visibility)
        sig_parts.append("struct")
        sig_parts.append(name)

        type_params = node.child_by_field_name("type_parameters")
        if type_params:
            sig_parts[-1] += self._get_node_text(type_params, content_bytes)

        signature = " ".join(sig_parts)

        return CodeSymbol(
            name=name,
            qualified_name=name,
            symbol_type="struct",
            content=source,
            language="rust",
            docstring=docstring,
            signature=signature,
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            visibility=visibility,
            metadata={
                "derives": derives,
                "generics": generics,
                "lifetimes": lifetimes,
                "where_clause": where_clause,
                "attributes": attributes,
            },
        )

    def _extract_enum(self, node, content_bytes: bytes) -> CodeSymbol:
        """Extract enum symbol.

        Args:
            node: Enum item node.
            content_bytes: Source code as bytes.

        Returns:
            CodeSymbol representing the enum.
        """
        cf = self._extract_common_fields(node, content_bytes)
        name, source, docstring, visibility, attributes = (
            cf.name, cf.source, cf.docstring, cf.visibility, cf.attributes
        )
        generics, lifetimes, where_clause = cf.generics, cf.lifetimes, cf.where_clause
        derives = self._parse_derives(attributes)

        # Extract variant names
        variants = []
        body = node.child_by_field_name("body")
        if body:
            for child in body.children:
                if child.type == "enum_variant":
                    variant_name = child.child_by_field_name("name")
                    if variant_name:
                        variants.append(self._get_node_text(variant_name, content_bytes))

        # Build signature
        sig_parts = []
        if visibility != "private":
            sig_parts.append(visibility)
        sig_parts.append("enum")
        sig_parts.append(name)

        type_params = node.child_by_field_name("type_parameters")
        if type_params:
            sig_parts[-1] += self._get_node_text(type_params, content_bytes)

        signature = " ".join(sig_parts)

        return CodeSymbol(
            name=name,
            qualified_name=name,
            symbol_type="enum",
            content=source,
            language="rust",
            docstring=docstring,
            signature=signature,
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            visibility=visibility,
            metadata={
                "derives": derives,
                "variants": variants,
                "generics": generics,
                "lifetimes": lifetimes,
                "where_clause": where_clause,
                "attributes": attributes,
            },
        )

    def _extract_trait(self, node, content_bytes: bytes) -> CodeSymbol:
        """Extract trait symbol.

        Args:
            node: Trait item node.
            content_bytes: Source code as bytes.

        Returns:
            CodeSymbol representing the trait.
        """
        cf = self._extract_common_fields(node, content_bytes)
        name, source, docstring, visibility, attributes = (
            cf.name, cf.source, cf.docstring, cf.visibility, cf.attributes
        )
        generics, lifetimes, where_clause = cf.generics, cf.lifetimes, cf.where_clause

        # Extract supertraits
        supertraits = []
        bounds = node.child_by_field_name("bounds")
        if bounds:
            supertraits.append(self._get_node_text(bounds, content_bytes))

        # Build signature
        sig_parts = []
        if visibility != "private":
            sig_parts.append(visibility)
        sig_parts.append("trait")
        sig_parts.append(name)

        type_params = node.child_by_field_name("type_parameters")
        if type_params:
            sig_parts[-1] += self._get_node_text(type_params, content_bytes)

        if supertraits:
            sig_parts.append(":")
            sig_parts.append(" + ".join(supertraits))

        signature = " ".join(sig_parts)

        return CodeSymbol(
            name=name,
            qualified_name=name,
            symbol_type="trait",
            content=source,
            language="rust",
            docstring=docstring,
            signature=signature,
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            visibility=visibility,
            metadata={
                "supertraits": supertraits,
                "generics": generics,
                "lifetimes": lifetimes,
                "where_clause": where_clause,
                "attributes": attributes,
            },
        )

    def _extract_trait_methods(
        self, node, content_bytes: bytes, symbols: list[CodeSymbol], trait_name: str
    ) -> None:
        """Extract methods from a trait definition.

        Args:
            node: Trait item node.
            content_bytes: Source code as bytes.
            symbols: List to append extracted symbols to.
            trait_name: Name of the containing trait.
        """
        body = node.child_by_field_name("body")
        if not body:
            return

        for child in body.children:
            if child.type == "function_item" or child.type == "function_signature_item":
                symbols.append(self._extract_function(child, content_bytes, trait_name))

    def _extract_impl(self, node, content_bytes: bytes) -> CodeSymbol:
        """Extract impl block symbol.

        Args:
            node: Impl item node.
            content_bytes: Source code as bytes.

        Returns:
            CodeSymbol representing the impl block.
        """
        cf = self._extract_common_fields(node, content_bytes, with_visibility=False)
        source, docstring, attributes = cf.source, cf.docstring, cf.attributes
        generics, lifetimes, where_clause = cf.generics, cf.lifetimes, cf.where_clause

        # Extract self type and trait (if impl Trait for Type)
        self_type = None
        trait_name = None

        type_node = node.child_by_field_name("type")
        if type_node:
            self_type = self._get_node_text(type_node, content_bytes)

        trait_node = node.child_by_field_name("trait")
        if trait_node:
            trait_name = self._get_node_text(trait_node, content_bytes)

        # Check for unsafe impl
        is_unsafe = False
        for child in node.children:
            if child.type == "unsafe":
                is_unsafe = True
                break

        # Build name and signature
        if trait_name:
            name = f"{trait_name} for {self_type}"
            signature = f"impl {trait_name} for {self_type}"
        else:
            name = self_type or "impl"
            signature = f"impl {self_type}"

        if is_unsafe:
            signature = "unsafe " + signature

        return CodeSymbol(
            name=name,
            qualified_name=name,
            symbol_type="impl",
            content=source,
            language="rust",
            docstring=docstring,
            signature=signature,
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            visibility="private",  # impl blocks don't have visibility
            metadata={
                "self_type": self_type,
                "trait": trait_name,
                "is_unsafe": is_unsafe,
                "generics": generics,
                "lifetimes": lifetimes,
                "where_clause": where_clause,
                "attributes": attributes,
            },
        )

    def _extract_impl_methods(
        self, node, content_bytes: bytes, symbols: list[CodeSymbol], impl_name: str
    ) -> None:
        """Extract methods from an impl block.

        Args:
            node: Impl item node.
            content_bytes: Source code as bytes.
            symbols: List to append extracted symbols to.
            impl_name: Name of the containing impl (usually the self type).
        """
        body = node.child_by_field_name("body")
        if not body:
            return

        # Use self_type as parent for methods
        type_node = node.child_by_field_name("type")
        parent_name = self._get_node_text(type_node, content_bytes) if type_node else impl_name

        for child in body.children:
            if child.type == "function_item":
                symbols.append(self._extract_function(child, content_bytes, parent_name))

    def _extract_type_alias(self, node, content_bytes: bytes) -> CodeSymbol:
        """Extract type alias symbol.

        Args:
            node: Type item node.
            content_bytes: Source code as bytes.

        Returns:
            CodeSymbol representing the type alias.
        """
        cf = self._extract_common_fields(node, content_bytes)
        name, source, docstring, visibility, attributes = (
            cf.name, cf.source, cf.docstring, cf.visibility, cf.attributes
        )
        generics, lifetimes, where_clause = cf.generics, cf.lifetimes, cf.where_clause

        # Get the aliased type
        type_node = node.child_by_field_name("type")
        aliased_type = self._get_node_text(type_node, content_bytes) if type_node else None

        # Build signature
        sig_parts = []
        if visibility != "private":
            sig_parts.append(visibility)
        sig_parts.append("type")
        sig_parts.append(name)

        type_params = node.child_by_field_name("type_parameters")
        if type_params:
            sig_parts[-1] += self._get_node_text(type_params, content_bytes)

        if aliased_type:
            sig_parts.append("=")
            sig_parts.append(aliased_type)

        signature = " ".join(sig_parts)

        return CodeSymbol(
            name=name,
            qualified_name=name,
            symbol_type="type_alias",
            content=source,
            language="rust",
            docstring=docstring,
            signature=signature,
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            visibility=visibility,
            metadata={
                "aliased_type": aliased_type,
                "generics": generics,
                "lifetimes": lifetimes,
                "where_clause": where_clause,
                "attributes": attributes,
            },
        )

    def _extract_const(self, node, content_bytes: bytes) -> CodeSymbol:
        """Extract const item symbol.

        Args:
            node: Const item node.
            content_bytes: Source code as bytes.

        Returns:
            CodeSymbol representing the constant.
        """
        cf = self._extract_common_fields(
            node, content_bytes, name_default="UNKNOWN", with_generics=False
        )
        name, source, docstring, visibility, attributes = (
            cf.name, cf.source, cf.docstring, cf.visibility, cf.attributes
        )

        # Get type
        type_node = node.child_by_field_name("type")
        const_type = self._get_node_text(type_node, content_bytes) if type_node else None

        # Build signature
        sig_parts = []
        if visibility != "private":
            sig_parts.append(visibility)
        sig_parts.append("const")
        sig_parts.append(f"{name}:")
        if const_type:
            sig_parts.append(const_type)

        signature = " ".join(sig_parts)

        return CodeSymbol(
            name=name,
            qualified_name=name,
            symbol_type="constant",
            content=source,
            language="rust",
            docstring=docstring,
            signature=signature,
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            visibility=visibility,
            metadata={
                "const_type": const_type,
                "attributes": attributes,
            },
        )

    def _extract_static(self, node, content_bytes: bytes) -> CodeSymbol:
        """Extract static item symbol.

        Args:
            node: Static item node.
            content_bytes: Source code as bytes.

        Returns:
            CodeSymbol representing the static.
        """
        cf = self._extract_common_fields(
            node, content_bytes, name_default="UNKNOWN", with_generics=False
        )
        name, source, docstring, visibility, attributes = (
            cf.name, cf.source, cf.docstring, cf.visibility, cf.attributes
        )

        # Check for mutable static
        is_mutable = False
        for child in node.children:
            if child.type == "mutable_specifier":
                is_mutable = True
                break

        # Get type
        type_node = node.child_by_field_name("type")
        static_type = self._get_node_text(type_node, content_bytes) if type_node else None

        # Build signature
        sig_parts = []
        if visibility != "private":
            sig_parts.append(visibility)
        sig_parts.append("static")
        if is_mutable:
            sig_parts.append("mut")
        sig_parts.append(f"{name}:")
        if static_type:
            sig_parts.append(static_type)

        signature = " ".join(sig_parts)

        return CodeSymbol(
            name=name,
            qualified_name=name,
            symbol_type="static",
            content=source,
            language="rust",
            docstring=docstring,
            signature=signature,
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            visibility=visibility,
            metadata={
                "static_type": static_type,
                "is_mutable": is_mutable,
                "attributes": attributes,
            },
        )

    def _extract_macro(self, node, content_bytes: bytes) -> CodeSymbol:
        """Extract macro_rules! definition.

        Args:
            node: Macro definition node.
            content_bytes: Source code as bytes.

        Returns:
            CodeSymbol representing the macro.
        """
        cf = self._extract_common_fields(
            node, content_bytes, with_visibility=False, with_generics=False
        )
        name, source, docstring, attributes = (
            cf.name, cf.source, cf.docstring, cf.attributes
        )

        # Check for macro_export
        is_exported = any("#[macro_export]" in attr for attr in attributes)

        signature = f"macro_rules! {name}"

        return CodeSymbol(
            name=name,
            qualified_name=name,
            symbol_type="macro",
            content=source,
            language="rust",
            docstring=docstring,
            signature=signature,
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            visibility="pub" if is_exported else "private",
            metadata={
                "is_exported": is_exported,
                "attributes": attributes,
            },
        )

    def _extract_module(self, node, content_bytes: bytes) -> CodeSymbol | None:
        """Extract inline module.

        Args:
            node: Mod item node.
            content_bytes: Source code as bytes.

        Returns:
            CodeSymbol representing the module, or None if it's an external mod declaration.
        """
        # Check if this is an inline module (has body) vs external mod declaration
        body = node.child_by_field_name("body")
        if not body:
            # External mod declaration like `mod utils;` - skip
            return None

        cf = self._extract_common_fields(
            node, content_bytes, with_generics=False
        )
        name, source, docstring, visibility, attributes = (
            cf.name, cf.source, cf.docstring, cf.visibility, cf.attributes
        )

        # Build signature
        sig_parts = []
        if visibility != "private":
            sig_parts.append(visibility)
        sig_parts.append("mod")
        sig_parts.append(name)

        signature = " ".join(sig_parts)

        return CodeSymbol(
            name=name,
            qualified_name=name,
            symbol_type="module",
            content=source,
            language="rust",
            docstring=docstring,
            signature=signature,
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            visibility=visibility,
            metadata={
                "attributes": attributes,
            },
        )

