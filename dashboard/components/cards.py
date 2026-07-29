"""
Enterprise KPI Metric Cards & Hero Banner Components.

Renders modern glassmorphism executive metric cards, AI copilot widgets, and linear-style headers.
"""

from typing import Optional
import streamlit as st


def render_kpi_card(
    value: str,
    label: str,
    border_color: str = "#6366F1",
    trend: Optional[str] = None,
    trend_type: str = "positive",
    subtext: Optional[str] = None,
):
    """
    Render an enterprise glassmorphic metric KPI card with trend indicator and elevation.
    """
    trend_html = ""
    if trend:
        css_class = "positive" if trend_type == "positive" else ("negative" if trend_type == "negative" else "neutral")
        icon = "↑" if trend_type == "positive" else ("↓" if trend_type == "negative" else "•")
        trend_html = f'<span class="kpi-trend {css_class}">{icon} {trend}</span>'

    subtext_html = f'<div class="kpi-subtext">{subtext}</div>' if subtext else ""

    card_html = f"""
    <div class="kpi-card" style="border-left: 4px solid {border_color};">
        <div class="kpi-card-header">
            <span class="kpi-label">{label}</span>
            {trend_html}
        </div>
        <div class="kpi-value">{value}</div>
        {subtext_html}
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)


def render_executive_header(
    title: str,
    subtitle: str,
    badge_text: str = "Enterprise AI Intelligence v2.4",
    status_online: bool = True,
):
    """
    Render Linear/Stripe style hero brand banner.
    """
    status_dot = '<span class="pulse-dot"></span> Online' if status_online else '<span style="color:#EF4444;">●</span> Offline'

    header_html = f"""
    <div class="brand-header">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <div class="brand-badge">
                ⚡ {badge_text}
            </div>
            <div style="font-size: 0.8rem; font-weight: 600; color: #34D399; display: flex; align-items: center; gap: 6px;">
                {status_dot}
            </div>
        </div>
        <h1 class="brand-title">{title}</h1>
        <p class="brand-subtitle">{subtitle}</p>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)


def render_ai_copilot_widget(
    query: str = "Analyze top retention priorities for Q3...",
    response: str = "AI Engine detected 142 high-value subscribers exhibiting early churn signals. Primary recommended action: 15% fiber upgrade discount + priority support queue.",
    confidence: float = 94.8,
):
    """
    Render futuristic AI Assistant Copilot card with streaming prompt styling.
    """
    widget_html = f"""
    <div class="ai-copilot-card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px;">
            <div class="ai-copilot-badge">
                ✨ AI Intelligence Copilot
            </div>
            <span style="font-size: 0.78rem; background: rgba(16, 185, 129, 0.15); color: #34D399; padding: 2px 10px; border-radius: 9999px; font-weight: 600; border: 1px solid rgba(16,185,129,0.3);">
                {confidence}% Confidence
            </span>
        </div>
        <div style="font-size: 0.85rem; color: #A5B4FC; font-weight: 600; margin-bottom: 6px;">
            💬 Prompt: "{query}"
        </div>
        <div style="font-size: 0.95rem; color: #F8FAFC; line-height: 1.6; background: rgba(0, 0, 0, 0.3); padding: 14px; border-radius: 8px; border-left: 3px solid #8B5CF6;">
            {response}
        </div>
        <div style="margin-top: 14px; display: flex; gap: 8px; flex-wrap: wrap;">
            <span class="ai-prompt-pill">🎯 Generate Campaign Plan</span>
            <span class="ai-prompt-pill">📈 Export Risk Cohort</span>
            <span class="ai-prompt-pill">🔍 Anomaly Inspection</span>
        </div>
    </div>
    """
    st.markdown(widget_html, unsafe_allow_html=True)


# Backward compatibility alias
render_ai_assistant_card = render_ai_copilot_widget

