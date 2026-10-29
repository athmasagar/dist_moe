# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU regressions for CuTeDSL kernel-name state ownership."""

import unittest
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

from dist_moe.kernels import _kernel_name_prefix as naming


@dataclass(frozen=True)
class _NameOptions:
    """Represent immutable naming settings on a decorated function."""

    name_prefix: str = ""
    remove_cutlass_symbol: bool = False
    keep_mangled_name: bool = True


class _DecoratedTarget:
    """Represent the function beneath a CuTeDSL jit wrapper."""


class _PrefixTarget:
    """Model the supported setter while exposing its shared DSL state."""

    def __init__(self, *, on_function: bool, dsl=None) -> None:
        """Create a wrapper using the selected naming storage contract."""
        self.on_function = on_function
        self.__wrapped__ = _DecoratedTarget()
        if dsl is not None:
            self.__wrapped__._dsl_object = dsl

    def set_name_prefix(self, prefix: str) -> None:
        """Record a prefix in the storage used by the selected DSL release."""
        if self.on_function:
            self.__wrapped__._cute_dsl_name_options = _NameOptions(name_prefix=prefix)
        else:
            self._name_prefix = prefix


class KernelNamePrefixTest(unittest.TestCase):
    """Check exact state restoration under both supported storage contracts."""

    def test_current_contract_restores_name_options(self) -> None:
        """Restore the exact options object, including non-prefix settings."""
        target = _PrefixTarget(on_function=True)
        previous = _NameOptions("external", True, False)
        target.__wrapped__._cute_dsl_name_options = previous
        with patch.object(naming, "NAME_OPTIONS_ON_FUNCTION", True):
            with naming.scoped_kernel_name_prefixes(((target, "distmoe"),)):
                self.assertEqual(
                    target.__wrapped__._cute_dsl_name_options.name_prefix, "distmoe"
                )
        self.assertIs(target.__wrapped__._cute_dsl_name_options, previous)

    def test_current_contract_removes_temporary_options(self) -> None:
        """Remove options introduced into a function that had no naming state."""
        target = _PrefixTarget(on_function=True)
        with patch.object(naming, "NAME_OPTIONS_ON_FUNCTION", True):
            with naming.scoped_kernel_name_prefixes(((target, "distmoe"),)):
                self.assertEqual(
                    target.__wrapped__._cute_dsl_name_options.name_prefix, "distmoe"
                )
        self.assertFalse(hasattr(target.__wrapped__, "_cute_dsl_name_options"))

    def test_legacy_contract_restores_wrapper_and_dsl(self) -> None:
        """Restore both a legacy wrapper and the DSL prefix copied by launch."""
        dsl = SimpleNamespace(_name_prefix="shared-before")
        target = _PrefixTarget(on_function=False, dsl=dsl)
        target._name_prefix = "wrapper-before"
        with patch.object(naming, "NAME_OPTIONS_ON_FUNCTION", False):
            with naming.scoped_kernel_name_prefixes(((target, "distmoe"),)):
                self.assertEqual(target._name_prefix, "distmoe")
                dsl._name_prefix = target._name_prefix
        self.assertEqual(target._name_prefix, "wrapper-before")
        self.assertEqual(dsl._name_prefix, "shared-before")

    def test_legacy_contract_removes_temporary_state(self) -> None:
        """Remove wrapper and DSL attributes that did not exist before use."""
        dsl = SimpleNamespace()
        target = _PrefixTarget(on_function=False, dsl=dsl)
        with patch.object(naming, "NAME_OPTIONS_ON_FUNCTION", False):
            with naming.scoped_kernel_name_prefixes(((target, "distmoe"),)):
                dsl._name_prefix = target._name_prefix
        self.assertFalse(hasattr(target, "_name_prefix"))
        self.assertFalse(hasattr(dsl, "_name_prefix"))

    def test_unrecognized_setter_rolls_back_options(self) -> None:
        """Reject unsupported recorded state without leaking temporary options."""
        target = _PrefixTarget(on_function=True)

        def set_unknown_state(prefix: str) -> None:
            """Model an unsupported setter storing a string instead of options."""
            target.__wrapped__._cute_dsl_name_options = prefix

        target.set_name_prefix = set_unknown_state
        with patch.object(naming, "NAME_OPTIONS_ON_FUNCTION", True):
            with self.assertRaisesRegex(RuntimeError, "did not record the prefix"):
                with naming.scoped_kernel_name_prefixes(((target, "distmoe"),)):
                    self.fail("an unrecognized setter must not enter the body")
        self.assertFalse(hasattr(target.__wrapped__, "_cute_dsl_name_options"))

    def test_failed_setter_restores_partial_state(self) -> None:
        """A setter exception restores original options, wrapper, and DSL state."""
        for on_function in (False, True):
            with self.subTest(on_function=on_function):
                dsl = SimpleNamespace(_name_prefix="shared-before")
                target = _PrefixTarget(on_function=on_function, dsl=dsl)
                target._name_prefix = "wrapper-before"
                previous = _NameOptions("options-before", True, False)
                target.__wrapped__._cute_dsl_name_options = previous

                def fail_after_mutating(
                    prefix: str,
                    *,
                    target: _PrefixTarget = target,
                    on_function: bool = on_function,
                    dsl: SimpleNamespace = dsl,
                ) -> None:
                    """Fail after partially writing naming state on all owners."""
                    target._name_prefix = "partial"
                    if on_function:
                        target.__wrapped__._cute_dsl_name_options = _NameOptions(
                            "partial"
                        )
                    dsl._name_prefix = "partial"
                    raise ValueError("setter failed")

                target.set_name_prefix = fail_after_mutating
                with patch.object(naming, "NAME_OPTIONS_ON_FUNCTION", on_function):
                    with self.assertRaisesRegex(ValueError, "setter failed"):
                        with naming.scoped_kernel_name_prefixes(((target, "distmoe"),)):
                            self.fail("a failed setter must not enter the body")
                self.assertEqual(target._name_prefix, "wrapper-before")
                self.assertEqual(dsl._name_prefix, "shared-before")
                self.assertIs(target.__wrapped__._cute_dsl_name_options, previous)

    def test_second_setter_failure_rolls_back_every_target(self) -> None:
        """Roll back the first successful setter when a later setter fails."""
        for on_function in (False, True):
            with self.subTest(on_function=on_function):
                first = _PrefixTarget(on_function=on_function)
                second = _PrefixTarget(on_function=on_function)

                def fail_after_mutating(
                    prefix: str,
                    *,
                    second: _PrefixTarget = second,
                ) -> None:
                    """Record a different prefix before failing the second setter."""
                    second.set_name_prefix = _PrefixTarget.set_name_prefix.__get__(
                        second
                    )
                    second.set_name_prefix("partial")
                    raise ValueError("second setter failed")

                second.set_name_prefix = fail_after_mutating
                with patch.object(naming, "NAME_OPTIONS_ON_FUNCTION", on_function):
                    with self.assertRaisesRegex(ValueError, "second setter failed"):
                        with naming.scoped_kernel_name_prefixes(
                            ((first, "first"), (second, "second"))
                        ):
                            self.fail("all setters must succeed before entering")
                for target in (first, second):
                    self.assertFalse(hasattr(target, "_name_prefix"))
                    self.assertFalse(
                        hasattr(target.__wrapped__, "_cute_dsl_name_options")
                    )

    def test_nested_current_scopes_restore_outer_identity(self) -> None:
        """Restore outer options by identity before restoring the original options."""
        target = _PrefixTarget(on_function=True)
        previous = _NameOptions("external", True, False)
        target.__wrapped__._cute_dsl_name_options = previous
        with patch.object(naming, "NAME_OPTIONS_ON_FUNCTION", True):
            with naming.scoped_kernel_name_prefixes(((target, "outer"),)):
                outer_options = target.__wrapped__._cute_dsl_name_options
                with naming.scoped_kernel_name_prefixes(((target, "inner"),)):
                    self.assertEqual(
                        target.__wrapped__._cute_dsl_name_options.name_prefix, "inner"
                    )
                self.assertIs(target.__wrapped__._cute_dsl_name_options, outer_options)
        self.assertIs(target.__wrapped__._cute_dsl_name_options, previous)

    def test_nested_legacy_scopes_restore_shared_prefix(self) -> None:
        """An inner legacy launch restores the outer wrapper and shared prefix."""
        dsl = SimpleNamespace(_name_prefix="shared-before")
        target = _PrefixTarget(on_function=False, dsl=dsl)
        target._name_prefix = "wrapper-before"
        with patch.object(naming, "NAME_OPTIONS_ON_FUNCTION", False):
            with naming.scoped_kernel_name_prefixes(((target, "outer"),)):
                dsl._name_prefix = "outer"
                with naming.scoped_kernel_name_prefixes(((target, "inner"),)):
                    dsl._name_prefix = "inner"
                self.assertEqual(target._name_prefix, "outer")
                self.assertEqual(dsl._name_prefix, "outer")
        self.assertEqual(target._name_prefix, "wrapper-before")
        self.assertEqual(dsl._name_prefix, "shared-before")

    def test_body_owned_current_options_are_preserved(self) -> None:
        """Retain a different prefix installed by the body instead of clobbering it."""
        target = _PrefixTarget(on_function=True)
        foreign = _NameOptions("external", True, False)
        with patch.object(naming, "NAME_OPTIONS_ON_FUNCTION", True):
            with naming.scoped_kernel_name_prefixes(((target, "distmoe"),)):
                target.__wrapped__._cute_dsl_name_options = foreign
        self.assertIs(target.__wrapped__._cute_dsl_name_options, foreign)

    def test_body_owned_legacy_prefixes_are_preserved(self) -> None:
        """Retain a different body-installed prefix on the wrapper and shared DSL."""
        dsl = SimpleNamespace(_name_prefix="shared-before")
        target = _PrefixTarget(on_function=False, dsl=dsl)
        with patch.object(naming, "NAME_OPTIONS_ON_FUNCTION", False):
            with naming.scoped_kernel_name_prefixes(((target, "distmoe"),)):
                target._name_prefix = "external"
                dsl._name_prefix = "external"
        self.assertEqual(target._name_prefix, "external")
        self.assertEqual(dsl._name_prefix, "external")

    def test_body_exception_restores_scope_owned_state(self) -> None:
        """A body exception restores installed state under either supported contract."""
        for on_function in (False, True):
            with self.subTest(on_function=on_function):
                dsl = SimpleNamespace(_name_prefix="shared-before")
                target = _PrefixTarget(on_function=on_function, dsl=dsl)
                target._name_prefix = "wrapper-before"
                previous = _NameOptions("options-before", True, False)
                target.__wrapped__._cute_dsl_name_options = previous
                with patch.object(naming, "NAME_OPTIONS_ON_FUNCTION", on_function):
                    with self.assertRaisesRegex(ValueError, "body failed"):
                        with naming.scoped_kernel_name_prefixes(((target, "distmoe"),)):
                            dsl._name_prefix = "distmoe"
                            raise ValueError("body failed")
                self.assertEqual(target._name_prefix, "wrapper-before")
                self.assertEqual(dsl._name_prefix, "shared-before")
                self.assertIs(target.__wrapped__._cute_dsl_name_options, previous)

    def test_shared_dsl_is_snapshotted_once(self) -> None:
        """Multiple wrappers sharing one DSL restore its single original prefix."""
        for on_function in (False, True):
            with self.subTest(on_function=on_function):
                dsl = SimpleNamespace(_name_prefix="shared-before")
                first = _PrefixTarget(on_function=on_function, dsl=dsl)
                second = _PrefixTarget(on_function=on_function, dsl=dsl)
                with patch.object(naming, "NAME_OPTIONS_ON_FUNCTION", on_function):
                    with naming.scoped_kernel_name_prefixes(
                        ((first, "first"), (second, "second"))
                    ):
                        dsl._name_prefix = "second"
                self.assertEqual(dsl._name_prefix, "shared-before")

    def test_lazy_materialization_is_one_way_and_scoped(self) -> None:
        """Materialize once while restoring the shared DSL prefix on each use."""
        target = _PrefixTarget(on_function=True)
        dsl = SimpleNamespace(_name_prefix="shared-before")
        initialized = []

        def initialize(decorated) -> None:
            """Model the one-way conversion from a DSL class to its shared object."""
            initialized.append(decorated)
            decorated._dsl_object = dsl
            del decorated._dsl_cls

        target.__wrapped__._dsl_cls = SimpleNamespace(_lazy_initialize_dsl=initialize)
        with patch.object(naming, "NAME_OPTIONS_ON_FUNCTION", True):
            for prefix in ("first", "second"):
                with naming.scoped_kernel_name_prefixes(((target, prefix),)):
                    self.assertIs(target.__wrapped__._dsl_object, dsl)
                    dsl._name_prefix = prefix
                self.assertEqual(dsl._name_prefix, "shared-before")
        self.assertEqual(initialized, [target.__wrapped__])
        self.assertFalse(hasattr(target.__wrapped__, "_dsl_cls"))
