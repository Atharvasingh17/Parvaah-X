"""
LLM Engine for PARVAAH-X Risk Intelligence
Uses Google Gemini API for natural language project risk summaries and Q&A.
Falls back to rule-based responses if API is unavailable.
"""
import os

# Load .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, will rely on env var or manual key

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

def _try_gemini(prompt: str, api_key: str) -> str:
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        return None

def _rule_based_summary(project: dict, risk_factors: list) -> str:
    name = project.get("Project_Name", "This project")
    sector = project.get("Sector", "Infrastructure")
    cost_risk = project.get("Cost_Risk_Score", 0) * 100
    time_risk = project.get("Time_Risk_Score", 0) * 100
    progress = project.get("Physical_Progress_Pct", 0)
    overrun = max(0, project.get("Revised_Cost_Cr", 0) - project.get("Original_Approved_Cost_Cr", 0))
    category = project.get("Risk_Category", "Medium")

    summary = f"**{name}** ({sector} sector) has been assessed as **{category} Risk** by the AI system. "

    if cost_risk > 60:
        summary += f"There is a {cost_risk:.0f}% probability of cost overrun, with an estimated additional expenditure of ₹{overrun:,.0f} crore over the original approved budget. "
    if time_risk > 60:
        summary += f"The project also faces a {time_risk:.0f}% chance of schedule delay beyond its planned completion date. "
    if progress < 30:
        summary += f"Physical progress stands at only {progress:.1f}%, which is critically low relative to time elapsed. "

    if risk_factors:
        summary += "Key contributing factors include: " + "; ".join(risk_factors[:3]) + ". "

    summary += "Immediate review and corrective action by the nodal ministry is recommended."
    return summary

def generate_risk_summary(project: dict, risk_factors: list, api_key: str = "") -> str:
    """
    Generate a natural language risk summary for a project.
    Uses Gemini API if key provided, else rule-based fallback.
    """
    key = api_key or GEMINI_API_KEY
    if key and len(key) > 10:
        prompt = f"""You are an expert infrastructure analyst for the Government of India, working under MoSPI.

Analyze the following project data and generate a crisp, professional risk assessment summary in 4-5 sentences.
Use formal language suitable for a government report. Mention specific numbers. End with a concrete recommendation.

Project Details:
- Name: {project.get('Project_Name')}
- Sector: {project.get('Sector')}
- Implementing Agency: {project.get('Implementing_Agency')}
- Original Budget: Rs. {project.get('Original_Approved_Cost_Cr', 0):,.0f} Crore
- Revised Cost: Rs. {project.get('Revised_Cost_Cr', 0):,.0f} Crore
- Physical Progress: {project.get('Physical_Progress_Pct', 0):.1f}%
- AI Cost Overrun Risk: {project.get('Cost_Risk_Score', 0)*100:.1f}%
- AI Time Delay Risk: {project.get('Time_Risk_Score', 0)*100:.1f}%
- Risk Category: {project.get('Risk_Category')}
- Climate Severity: {project.get('Climate_Issue_Severity', 0):.2f}
- Geopolitical Risk: {project.get('Geopolitical_War_Impact', 0):.2f}
- Supply Chain Risk: {project.get('Supply_Chain_Disruption', 0):.2f}
- Expenditure-Progress Ratio: {project.get('Expenditure_Progress_Ratio', 1):.2f}

Top AI-identified risk factors: {', '.join(risk_factors[:5]) if risk_factors else 'None identified'}

Write the assessment now:"""

        result = _try_gemini(prompt, key)
        if result:
            return result

    return _rule_based_summary(project, risk_factors)


def answer_project_question(project: dict, question: str, api_key: str = "") -> str:
    """
    Answer a user question about a specific project using LLM or rule-based logic.
    """
    key = api_key or GEMINI_API_KEY
    if key and len(key) > 10:
        prompt = f"""You are an AI assistant for PARVAAH-X, the Government of India's infrastructure project monitoring system.
Answer the following question about a project concisely and accurately in 2-4 sentences. Use data from the project details.
If unsure, say so — do not hallucinate numbers.

Project Details:
- Name: {project.get('Project_Name')}
- Sector: {project.get('Sector')}
- Agency: {project.get('Implementing_Agency')}
- Budget: Rs. {project.get('Original_Approved_Cost_Cr', 0):,.0f} Cr | Revised: Rs. {project.get('Revised_Cost_Cr', 0):,.0f} Cr
- Progress: {project.get('Physical_Progress_Pct', 0):.1f}%
- Cost Risk: {project.get('Cost_Risk_Score', 0)*100:.1f}% | Time Risk: {project.get('Time_Risk_Score', 0)*100:.1f}%
- Risk Level: {project.get('Risk_Category')}
- Climate Issue: {project.get('Climate_Issue_Severity', 0):.2f}
- Geopolitical Impact: {project.get('Geopolitical_War_Impact', 0):.2f}
- Supply Chain: {project.get('Supply_Chain_Disruption', 0):.2f}
- Start Date: {project.get('Start_Date')} | Planned Completion: {project.get('Planned_Completion_Date')}
- Current Expected Completion: {project.get('Current_Completion_Date')}

Question: {question}

Answer:"""
        result = _try_gemini(prompt, key)
        if result:
            return result

    # Rule-based fallback
    q_lower = question.lower()
    if "delay" in q_lower or "time" in q_lower or "complete" in q_lower or "finish" in q_lower:
        return (f"Based on AI analysis, {project.get('Project_Name')} has a **{project.get('Time_Risk_Score', 0)*100:.1f}%** probability of schedule delay. "
                f"It is expected to complete by {project.get('Current_Completion_Date', 'N/A')} vs planned {project.get('Planned_Completion_Date', 'N/A')}. "
                f"Key delay factors include supply chain disruption ({project.get('Supply_Chain_Disruption', 0):.2f}) and climate risks ({project.get('Climate_Issue_Severity', 0):.2f}).")
    elif "cost" in q_lower or "overrun" in q_lower or "budget" in q_lower or "money" in q_lower or "paisa" in q_lower:
        overrun = max(0, project.get('Revised_Cost_Cr', 0) - project.get('Original_Approved_Cost_Cr', 0))
        return (f"The original approved budget for {project.get('Project_Name')} was ₹{project.get('Original_Approved_Cost_Cr', 0):,.0f} Cr. "
                f"The revised cost is ₹{project.get('Revised_Cost_Cr', 0):,.0f} Cr, indicating a cost overrun of ₹{overrun:,.0f} Cr ({overrun/max(1,project.get('Original_Approved_Cost_Cr',1))*100:.1f}%). "
                f"AI model assigns a {project.get('Cost_Risk_Score', 0)*100:.1f}% probability of further cost escalation.")
    elif "progress" in q_lower or "kitna" in q_lower or "complete" in q_lower:
        return f"Physical progress for {project.get('Project_Name')} stands at **{project.get('Physical_Progress_Pct', 0):.1f}%** of total work completed."
    elif "risk" in q_lower or "danger" in q_lower:
        return (f"{project.get('Project_Name')} is classified as **{project.get('Risk_Category')} Risk**. "
                f"Overall risk score: {(project.get('Cost_Risk_Score', 0)*0.5 + project.get('Time_Risk_Score', 0)*0.5)*100:.1f}%. "
                f"Primary risk drivers are geopolitical impact ({project.get('Geopolitical_War_Impact', 0):.2f}), supply chain disruption ({project.get('Supply_Chain_Disruption', 0):.2f}), and climate severity ({project.get('Climate_Issue_Severity', 0):.2f}).")
    else:
        return (f"I don't have enough specific data to answer that precisely. Based on available data, "
                f"{project.get('Project_Name')} is a {project.get('Sector')} project under {project.get('Implementing_Agency')}, "
                f"currently at {project.get('Physical_Progress_Pct', 0):.1f}% completion with a {project.get('Risk_Category')} risk profile. "
                f"Please ask about cost, timeline, progress, or risk factors for a detailed answer.")
