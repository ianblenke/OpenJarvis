"""Typed test fakes for anti-mocking test strategy.

Instead of using unittest.mock.MagicMock, use these real implementations
that satisfy the same protocols/interfaces. This catches interface drift
and ensures tests exercise real code paths.
"""
