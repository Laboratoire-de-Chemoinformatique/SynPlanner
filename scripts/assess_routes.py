#!/usr/bin/env python3
"""Compatibility entry point for forward-only assessment of saved routes."""

from scripts.assess_forward_routes import assess_forward_collection, main

assess_collection = assess_forward_collection

__all__ = ["assess_collection", "main"]

if __name__ == "__main__":
    main()
