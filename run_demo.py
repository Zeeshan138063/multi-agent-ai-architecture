#!/usr/bin/env python3
"""
Simple script to run the Modular Agentic AI demo from the project root.
This handles the Python path setup automatically.
"""

import sys
import os
import asyncio

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import and run the demo
from modular_agentic_ai.examples.modular_demo import main

if __name__ == "__main__":
    print("🚀 Starting Modular Agentic AI Architecture Demo...")
    print("This may take a moment to initialize all components.\n")
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()