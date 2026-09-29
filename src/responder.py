"""
Automated Customer Response Generator (Bonus Generative AI Task).

Strictly 100% offline and API-free.
Uses category-based policy mappings, entity/keyword heuristics, and priority SLAs
to generate empathetic, actionable, and context-specific customer responses.
"""

from typing import Dict, Any, Optional


# Category-specific template strategies and immediate guidance
CATEGORY_RESPONSES = {
    "Login Issue": {
        "greeting": "Hello, thank you for reaching out regarding your account access.",
        "core_advice": (
            "We understand you are experiencing difficulty signing in. "
            "Please use the 'Forgot Password' link on the login page to initiate a secure password reset. "
            "Also ensure that your browser cookies and cache are cleared, and verify that Caps Lock is disabled."
        ),
        "next_step": "If you do not receive the reset email within 5 minutes, our identity support team will manually assist you.",
        "recommended_action": "Self-service password reset or account verification dispatch.",
    },
    "Application Error": {
        "greeting": "Hello, thank you for reporting this application issue.",
        "core_advice": (
            "We sincerely apologize for the disruption caused by this unexpected error. "
            "Our engineering team has logged this incident. If possible, taking a screenshot of the error code "
            "and trying to refresh the session in a private/incognito window can help isolate cache conflicts."
        ),
        "next_step": "A technical specialist is reviewing the server telemetry and will update you shortly.",
        "recommended_action": "Engineering ticket triage and server log inspection.",
    },
    "Report": {
        "greeting": "Hello, thank you for requesting reporting assistance.",
        "core_advice": (
            "Your report request has been logged. You can access on-demand data exports directly "
            "from the 'Analytics & Reports' tab in your user dashboard under the 'Export CSV/PDF' section."
        ),
        "next_step": "If you require custom filters or scheduled recurring deliveries, our BI specialist will configure them for you.",
        "recommended_action": "BI team report export or custom query generation.",
    },
    "Account Update": {
        "greeting": "Hello, thank you for contacting us regarding your profile details.",
        "core_advice": (
            "We have received your update request. For your security, sensitive modifications "
            "(such as primary mobile number or legal billing address) require a quick secondary verification code "
            "sent to your registered contact channel."
        ),
        "next_step": "Please check your registered inbox for a verification link to confirm this update.",
        "recommended_action": "Account security verification & credential update.",
    },
    "Performance": {
        "greeting": "Hello, thank you for notifying us about system responsiveness.",
        "core_advice": (
            "We recognize that fast loading times are critical to your workflow. "
            "Our infrastructure operations team is actively monitoring server load, database query latency, "
            "and CDN distribution to resolve this slowdown."
        ),
        "next_step": "Performance telemetry is currently under inspection; expect normal speeds to be restored swiftly.",
        "recommended_action": "Infrastructure scaling and database query profiling.",
    },
    "Payment Issue": {
        "greeting": "Hello, thank you for contacting our billing and finance desk.",
        "core_advice": (
            "We apologize for the inconvenience regarding your payment transaction. "
            "If your bank was debited while the transaction status shows pending, bank holds typically auto-reconcile "
            "within 24 to 48 hours without duplicate debits."
        ),
        "next_step": "Our billing team is cross-referencing your transaction reference with our payment gateway.",
        "recommended_action": "Payment gateway reconciliation and merchant ledger verification.",
    },
    "Access Issue": {
        "greeting": "Hello, thank you for contacting us regarding system permissions.",
        "core_advice": (
            "We have received your request for elevated access or permission adjustment. "
            "System access controls are managed via Role-Based Access Control (RBAC). "
            "Please confirm if your department manager has approved this role delegation."
        ),
        "next_step": "Our security administrator has been notified to provision the necessary workspace role.",
        "recommended_action": "RBAC permission provisioning and security verification.",
    },
    "Data Issue": {
        "greeting": "Hello, thank you for notifying our data integrity team.",
        "core_advice": (
            "We take data consistency and accurate recordkeeping very seriously. "
            "Our database administrators have been alerted to review historical revision audits and recent sync events."
        ),
        "next_step": "We are verifying backup snapshots to guarantee no records are lost or duplicated.",
        "recommended_action": "Database audit trail inspection and snapshot verification.",
    },
}

# Priority-to-SLA mapping
PRIORITY_SLA = {
    "Critical": "Within 1 hour (Critical Priority SLA)",
    "High": "Within 4 business hours (High Priority SLA)",
    "Medium": "Within 24 business hours (Standard SLA)",
    "Low": "Within 48 business hours (General Inquiries SLA)",
}


def generate_automated_response(
    category: str,
    ticket_description: str = "",
    customer_name: Optional[str] = None,
    priority: str = "Medium",
    confidence: Optional[float] = None,
    is_low_confidence: bool = False,
    confidence_threshold: float = 40.0,
) -> Dict[str, Any]:
    """
    Generates a professional, automated support response for the customer.
    Runs completely offline with zero external API calls.
    
    If is_low_confidence is True, activates the operational fallback response:
    instead of sending an uncertain category action, it politely requests
    clarification and routes the ticket to manual human triage.
    """
    salutation = f"Dear {customer_name}," if customer_name else "Dear Customer,"
    sla_time = PRIORITY_SLA.get(priority, "Within 24 hours")

    # Fallback response for ambiguous / low-confidence predictions
    if is_low_confidence:
        desc_snippet = ticket_description.strip()
        if len(desc_snippet) > 80:
            desc_snippet = desc_snippet[:77] + "..."

        core_advice = (
            f"Thank you for contacting customer support. We have received your inquiry:\n"
            f"\"{desc_snippet}\"\n\n"
            "Because this description does not contain sufficient technical details for automated routing, "
            "your ticket has been placed into our Tier-1 Priority Triage Queue for manual evaluation by a support engineer."
        )
        next_step = (
            "To help us resolve your issue promptly, please reply to this ticket with additional details:\n"
            "  • Any exact error messages, status codes (e.g., 500, 403), or screenshots\n"
            "  • The specific screen or action you were performing\n"
            "  • Your operating system and browser version"
        )
        conf_str = f"{confidence:.1f}%" if confidence is not None else "Unknown"
        confidence_disclaimer = (
            f"\n\n[Automated Triage Dispatch - Low Confidence Fallback Activated: "
            f"Model suggestion: '{category}' at {conf_str} (below {confidence_threshold:.1f}% threshold)]"
        )
        response_text = (
            f"{salutation}\n\n"
            f"{core_advice}\n\n"
            f"{next_step}\n\n"
            f"Estimated Resolution Timeframe: {sla_time}.\n\n"
            f"Best regards,\nCustomer Support Operations Team"
            f"{confidence_disclaimer}"
        )
        return {
            "category": category,
            "response_text": response_text,
            "recommended_action": "Route to Tier-1 Support Desk for manual triage. Request error details/screenshots from customer.",
            "estimated_sla": sla_time,
        }

    # Standard confident response dispatch
    info = CATEGORY_RESPONSES.get(
        category,
        {
            "greeting": "Hello, thank you for submitting your support ticket.",
            "core_advice": "We have received your query and routed it to our specialized customer support team.",
            "next_step": "A support agent will examine your ticket details and follow up with a resolution.",
            "recommended_action": "General support ticket triage.",
        },
    )

    # Keyword enrichment
    desc_lower = ticket_description.lower()
    custom_note = ""
    if "urgent" in desc_lower or "immediately" in desc_lower or "asap" in desc_lower:
        custom_note = "\n\nNotice: We noted your request is time-sensitive and have flagged it for expedited review."
    elif "password" in desc_lower:
        custom_note = "\n\nTip: For quick resolution, ensure you check your Spam/Junk folder for password reset emails."
    elif "invoice" in desc_lower or "tax" in desc_lower:
        custom_note = "\n\nTip: Invoices for completed billing cycles are downloadable under Settings > Billing."

    confidence_disclaimer = ""
    if confidence is not None:
        confidence_disclaimer = f"\n\n[Automated System Dispatch - Category: {category} ({confidence:.1f}% confidence)]"

    response_text = (
        f"{salutation}\n\n"
        f"{info['greeting']}\n\n"
        f"{info['core_advice']}\n\n"
        f"{info['next_step']}{custom_note}\n\n"
        f"Estimated Resolution Timeframe: {sla_time}.\n\n"
        f"Best regards,\nCustomer Support Operations Team"
        f"{confidence_disclaimer}"
    )

    return {
        "category": category,
        "response_text": response_text,
        "recommended_action": info["recommended_action"],
        "estimated_sla": sla_time,
    }


if __name__ == "__main__":
    sample = generate_automated_response(
        category="Login Issue",
        ticket_description="I forgot my password and cannot sign into the application",
        customer_name="John Doe",
        priority="High",
        confidence=94.5,
    )
    print("=" * 60)
    print("SAMPLE GENERATED AUTOMATED RESPONSE (OFFLINE)")
    print("=" * 60)
    print(sample["response_text"])
    print(f"\nRecommended Action: {sample['recommended_action']}")
    print(f"SLA: {sample['estimated_sla']}")
