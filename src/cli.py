"""
Command Line Interface (CLI) for Customer Support Ticket Classification.
Assignment Task 6 - Option A.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import predict_ticket


def print_banner():
    banner = """
========================================================================
           CUSTOMER SUPPORT TICKET INTELLIGENCE SYSTEM
             (Task 6 - Interactive Command Line Interface)
========================================================================
  • 100% Offline Machine Learning (Zero External APIs)
  • TF-IDF Vectorization + Multinomial Logistic Regression
  • Automated Resolution Guidance (Bonus Generative AI Task)
------------------------------------------------------------------------
Commands:
  - Type any support ticket description and press Enter.
  - Type 'samples' to view quick test examples.
  - Type 'exit' or 'quit' to terminate.
========================================================================
"""
    print(banner)


def show_samples():
    print("\n[Sample Ticket Descriptions to Try]:")
    print(" 1. I am unable to login because my password is not working.")
    print(" 2. The application crashes with an unhandled exception when saving.")
    print(" 3. Please export the quarterly sales report for my team.")
    print(" 4. I need to update my registered email and contact number.")
    print(" 5. Database search queries are taking too long and timing out.")
    print(" 6. Transaction completed but money was deducted twice.")
    print(" 7. Access denied when clicking on the administration panel.")
    print(" 8. Some historical order records have disappeared from my view.\n")


def run_interactive_cli():
    print_banner()
    while True:
        try:
            user_input = input("\nEnter Ticket Description > ").strip()
            if not user_input:
                continue

            if user_input.lower() in ("exit", "quit", "q"):
                print("\nExiting Customer Support Classifier. Goodbye!\n")
                break

            if user_input.lower() == "samples":
                show_samples()
                continue

            result = predict_ticket(user_input)

            print("\n" + "-" * 60)
            print(f"PREDICTED CATEGORY : [{result['predicted_category'].upper()}]")
            print(f"CONFIDENCE SCORE   : {result['confidence']:.1f}%")
            print(f"RECOMMENDED ACTION : {result['recommended_action']}")
            print(f"ESTIMATED SLA      : {result['estimated_sla']}")
            print("-" * 60)
            print("\nProbability Breakdown:")
            for cat, prob in list(result["all_probabilities"].items())[:4]:
                bar = "█" * int(prob / 5)
                print(f"  {cat:18s} : {prob:>5.1f}% | {bar}")

            print("\n" + "~" * 60)
            print("AUTOMATED CUSTOMER RESPONSE (OFFLINE):")
            print(result["suggested_response"])
            print("~" * 60)

        except (KeyboardInterrupt, EOFError):
            print("\nSession interrupted. Exiting...")
            break


if __name__ == "__main__":
    run_interactive_cli()
