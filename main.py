# --- RFP Agentic AI Demo ---
# Simple version for beginners (runs directly in VS Code)


import os
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Step 1: Connect to OpenAI with your key

    """Summarizes the RFP and extracts requirements"""
    prompt = f"""
    You are a Sales Agent.
    Read this RFP text and summarize it in 4 points:
    1. Product required
    2. Quantity
    3. Any technical specifications
    4. Testing / acceptance requirements
    
    RFP Text:
    {rfp_text}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    summary = response.choices[0].message.content
    print("\n--- SALES AGENT SUMMARY ---")
    print(summary)
    return summary


# ---------- TECHNICAL AGENT ----------
def technical_agent(summary):
    """Matches RFP requirements with internal SKUs"""
    sample_products = [
        {"SKU": "CBL101", "Type": "11kV Cable", "Insulation": "XLPE"},
        {"SKU": "CBL102", "Type": "33kV Cable", "Insulation": "PVC"},
        {"SKU": "CBL103", "Type": "1.1kV Cable", "Insulation": "PVC"}
    ]

    prompt = f"""
    You are a Technical Agent.
    Based on the RFP summary below, choose which product SKU is the best match.
    Explain why.

    RFP Summary:
    {summary}

    Product Catalog:
    {sample_products}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    match = response.choices[0].message.content
    print("\n--- TECHNICAL AGENT MATCH ---")
    print(match)
    return match


# ---------- PRICING AGENT ----------
def pricing_agent(technical_output):
    """Estimates pricing based on technical details"""
    prompt = f"""
    You are a Pricing Agent.
    Given the technical match below, estimate:
    1. Approx price (in INR)
    2. Additional testing or delivery cost
    3. Total cost summary

    Technical Details:
    {technical_output}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    pricing = response.choices[0].message.content
    print("\n--- PRICING AGENT ESTIMATE ---")
    print(pricing)
    return pricing


# ---------- MASTER FLOW ----------
if __name__ == "__main__":
    rfp_text = """
    Tender for supply of 500 meters of 11kV XLPE insulated copper cable conforming to IS 7098 Part II.
    Testing to be done at site before acceptance. Submission deadline: November 25, 2025.
    """

    summary = sales_agent(rfp_text)
    tech_output = technical_agent(summary)
    pricing = pricing_agent(tech_output)

    print("\n==============================")
    print("FINAL RFP RESPONSE (SIMULATED)")
    print("==============================")
    print(f"\nRFP Summary:\n{summary}")
    print(f"\nTechnical Match:\n{tech_output}")
    print(f"\nPricing Details:\n{pricing}")
