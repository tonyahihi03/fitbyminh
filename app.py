import torch
import streamlit as st
import time
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

st.set_page_config(page_title="FitByMinh", page_icon=None, layout="wide", initial_sidebar_state="collapsed")

# ============================================================
# DARK THEME — FitByMinh Design System
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --bg-base:       #0C0C0F;
    --bg-panel:      #0F0F14;
    --bg-surface:    #13131A;
    --bg-elevated:   #18181F;
    --bg-overlay:    #1E1E27;

    --border-subtle:  rgba(255,255,255,0.04);
    --border-default: rgba(255,255,255,0.07);
    --border-strong:  rgba(255,255,255,0.13);
    --border-accent:  rgba(249,115,22,0.5);
    --border-ai:      rgba(139,92,246,0.45);

    --text-primary:   #FAFAFA;
    --text-secondary: #D4D4D8;
    --text-muted:     #A1A1AA;
    --text-faint:     #71717D;
    --text-disabled:  #52525E;

    --accent:         #F97316;
    --accent-hover:   #FB923C;
    --accent-muted:   rgba(249,115,22,0.15);
    --accent-subtle:  rgba(249,115,22,0.08);

    --ai:             #8B5CF6;
    --ai-hover:       #A78BFA;
    --ai-muted:       rgba(139,92,246,0.15);
    --ai-subtle:      rgba(139,92,246,0.08);

    --success: #22C55E;
    --warning: #FBBF24;
    --danger:  #EF4444;

    --font-display: 'Sora', sans-serif;
    --font-body:    'Inter', sans-serif;
    --font-mono:    'JetBrains Mono', monospace;

    --ease-spring: cubic-bezier(0.16, 1, 0.3, 1);
    --shadow-card: 0 1px 3px rgba(0,0,0,0.4), 0 0 0 1px rgba(255,255,255,0.05);
    --shadow-lg:   0 8px 32px rgba(0,0,0,0.6), 0 0 0 1px rgba(255,255,255,0.07);
    --glow-accent: 0 0 30px rgba(249,115,22,0.18);
}

* { font-family: var(--font-body); box-sizing: border-box; -webkit-font-smoothing: antialiased; }

.stApp {
    background: var(--bg-base);
    color: var(--text-primary);
    background-image:
        radial-gradient(ellipse 600px 400px at 80% -100px, rgba(249,115,22,0.08), transparent),
        radial-gradient(ellipse 500px 300px at 10% 110%, rgba(139,92,246,0.05), transparent);
    background-attachment: fixed;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }

::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.08); border-radius: 99px; }
::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.14); }

/* ===== NAV ===== */
.nav {
    display: flex; align-items: center; justify-content: space-between;
    padding: 14px 32px;
    border-bottom: 1px solid var(--border-default);
    background: rgba(15,15,20,0.7);
    backdrop-filter: blur(12px);
    position: sticky; top: 0; z-index: 100;
}
.nav-logo {
    font-family: var(--font-display);
    font-size: 17px; font-weight: 800; letter-spacing: -0.02em;
    color: var(--text-primary);
    display: flex; align-items: center; gap: 10px;
}
.nav-mark {
    width: 26px; height: 26px; border-radius: 7px;
    background: linear-gradient(135deg, var(--accent) 0%, #C2410C 100%);
    display: flex; align-items: center; justify-content: center;
    color: white; font-weight: 800; font-size: 13px;
    box-shadow: var(--glow-accent);
}
.nav-logo span { color: var(--accent); }
.nav-status {
    display: flex; align-items: center; gap: 8px;
    font-size: 11px; color: var(--text-faint);
    font-weight: 500; letter-spacing: 0.04em; text-transform: uppercase;
}
.status-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: var(--success); box-shadow: 0 0 8px var(--success);
    animation: pulse 2s var(--ease-spring) infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; } 50% { opacity: 0.5; }
}

/* ===== HERO ===== */
.hero { padding: 64px 48px 40px; max-width: 1200px; margin: 0 auto; }
.hero-eyebrow {
    display: inline-flex; align-items: center; gap: 8px;
    font-size: 11px; letter-spacing: 0.12em; text-transform: uppercase;
    color: var(--accent); font-weight: 600; margin-bottom: 24px;
    background: var(--accent-subtle); border: 1px solid var(--accent-muted);
    padding: 6px 12px; border-radius: 999px;
}
.hero-eyebrow::before {
    content: ''; width: 6px; height: 6px; border-radius: 50%; background: var(--accent);
}
.hero-title {
    font-family: var(--font-display);
    font-size: 56px; line-height: 1.05;
    font-weight: 800; letter-spacing: -0.03em;
    color: var(--text-primary);
    margin: 0 0 18px; max-width: 760px;
}
.hero-title em {
    font-style: normal;
    background: linear-gradient(135deg, var(--accent) 0%, var(--ai) 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-body {
    font-size: 16px; color: var(--text-muted);
    line-height: 1.6; max-width: 580px;
}

/* ===== FORM ===== */
.form-wrap { padding: 32px 48px; max-width: 1200px; margin: 0 auto; }
.section-label {
    font-size: 10px; font-weight: 600; letter-spacing: 0.12em;
    text-transform: uppercase; color: var(--text-disabled);
    margin-bottom: 18px; padding-bottom: 12px;
    border-bottom: 1px solid var(--border-default);
    display: flex; align-items: center; gap: 8px;
}
.section-label::before {
    content: ''; width: 4px; height: 4px; background: var(--accent); border-radius: 50%;
}

/* Streamlit overrides */
.stSelectbox label, .stNumberInput label, .stSlider label, .stTextInput label {
    font-size: 11px !important;
    font-weight: 500 !important;
    color: var(--text-faint) !important;
    text-transform: none !important;
    letter-spacing: 0 !important;
    font-family: var(--font-body) !important;
    margin-bottom: 6px !important;
}
.stSelectbox > div > div, .stTextInput > div > div > input {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-default) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
    font-size: 13px !important;
    font-family: var(--font-body) !important;
    transition: border-color 200ms var(--ease-spring) !important;
}
.stSelectbox > div > div:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(249,115,22,0.15) !important;
}
.stNumberInput > div > div > input {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-default) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
    font-size: 13px !important;
    font-family: var(--font-mono) !important;
    padding: 10px 12px !important;
    transition: border-color 200ms var(--ease-spring) !important;
}
.stNumberInput > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(249,115,22,0.15) !important;
}
.stSlider > div > div > div > div { background: var(--accent) !important; }
.stSlider [data-baseweb="slider"] [role="slider"] {
    background: var(--accent) !important;
    box-shadow: 0 0 0 4px rgba(249,115,22,0.2) !important;
}

/* ===== STATS STRIP ===== */
.stats-strip {
    background: var(--bg-surface);
    border: 1px solid var(--border-default);
    border-radius: 12px;
    padding: 18px 24px;
    margin: 20px 0 28px;
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 0;
    box-shadow: var(--shadow-card);
}
.stat-cell {
    padding: 0 18px;
    border-right: 1px solid var(--border-subtle);
    text-align: left;
}
.stat-cell:first-child { padding-left: 0; }
.stat-cell:last-child { border-right: none; padding-right: 0; }
.stat-num {
    font-family: var(--font-mono);
    font-size: 26px;
    font-weight: 600;
    letter-spacing: -0.02em;
    color: var(--text-primary);
    line-height: 1.1;
}
.stat-num.accent { color: var(--accent); }
.stat-lbl {
    font-size: 10px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--text-faint);
    margin-top: 6px;
    font-weight: 500;
}
.stat-tag { font-size: 11px; font-weight: 500; margin-top: 4px; }
.stat-tag.healthy { color: var(--success); }
.stat-tag.warning { color: var(--warning); }
.stat-tag.danger  { color: var(--danger); }

/* ===== TRAINER NOTE ===== */
.trainer-note {
    background: linear-gradient(135deg, var(--bg-surface) 0%, var(--bg-elevated) 100%);
    border: 1px solid var(--border-default);
    border-left: 3px solid var(--accent);
    border-radius: 12px;
    padding: 22px 26px;
    margin: 16px 0 28px;
    box-shadow: var(--shadow-card);
}
.note-label {
    font-size: 10px; letter-spacing: 0.12em; text-transform: uppercase;
    color: var(--accent); font-weight: 600; margin-bottom: 10px;
    display: flex; align-items: center; gap: 8px;
}
.note-main {
    font-family: var(--font-display);
    font-size: 17px; line-height: 1.5;
    color: var(--text-primary); font-weight: 600;
    letter-spacing: -0.01em;
}
.note-sub {
    font-size: 13px; color: var(--text-muted);
    margin-top: 10px; line-height: 1.6;
}

/* ===== PRIMARY BUTTON ===== */
.stButton > button {
    background: var(--accent) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: var(--font-body) !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    padding: 14px 32px !important;
    width: 100% !important;
    text-transform: none !important;
    transition: all 200ms var(--ease-spring) !important;
    box-shadow: 0 0 20px rgba(249,115,22,0.25) !important;
}
.stButton > button:hover {
    background: var(--accent-hover) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 0 32px rgba(249,115,22,0.4) !important;
}
.stButton > button:active { transform: translateY(0) scale(0.98) !important; }

/* ===== PLAN OUTPUT ===== */
.plan-section {
    background: var(--bg-base);
    border-top: 1px solid var(--border-default);
    padding: 48px;
    margin-top: 24px;
}
.plan-wrap { max-width: 1200px; margin: 0 auto; }
.plan-header {
    display: flex; align-items: flex-end; justify-content: space-between;
    padding-bottom: 24px;
    border-bottom: 1px solid var(--border-default);
    margin-bottom: 32px;
    flex-wrap: wrap; gap: 16px;
}
.plan-title-block { }
.plan-title {
    font-family: var(--font-display);
    font-size: 36px; letter-spacing: -0.025em;
    color: var(--text-primary); font-weight: 700;
    margin-bottom: 6px;
}
.plan-meta {
    font-size: 12px; color: var(--text-muted);
    letter-spacing: 0.04em;
    display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
}
.meta-pill {
    background: var(--bg-surface); border: 1px solid var(--border-default);
    border-radius: 999px; padding: 4px 10px;
    font-size: 11px; color: var(--text-muted); font-weight: 500;
}
.meta-pill.accent {
    background: var(--accent-subtle);
    border-color: var(--accent-muted);
    color: var(--accent-hover);
}

.adjust-banner {
    background: var(--ai-subtle);
    border: 1px solid var(--ai-muted);
    border-radius: 10px; padding: 14px 18px;
    margin-bottom: 28px;
    font-size: 13px; color: var(--text-secondary);
    line-height: 1.55;
    display: flex; align-items: flex-start; gap: 12px;
}
.adjust-banner-icon {
    width: 32px; height: 32px; border-radius: 8px;
    background: var(--ai-muted); border: 1px solid var(--ai);
    display: flex; align-items: center; justify-content: center;
    color: var(--ai-hover); font-weight: 700; flex-shrink: 0;
    font-family: var(--font-mono);
}
.adjust-banner strong { color: var(--text-primary); }

.subsection-label {
    font-family: var(--font-display);
    font-size: 12px; font-weight: 600; letter-spacing: 0.08em;
    text-transform: uppercase; color: var(--text-faint);
    margin: 24px 0 14px;
    padding-bottom: 10px;
    border-bottom: 1px solid var(--border-default);
    display: flex; align-items: center; gap: 8px;
}

/* ===== DAY GRID ===== */
.plan-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 14px;
    margin-bottom: 8px;
}
.day-card {
    background: var(--bg-surface);
    border: 1px solid var(--border-default);
    border-radius: 12px;
    overflow: hidden;
    transition: all 200ms var(--ease-spring);
    box-shadow: var(--shadow-card);
}
.day-card:hover {
    border-color: var(--border-strong);
    transform: translateY(-2px);
}
.day-header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 14px 18px;
    border-bottom: 1px solid var(--border-subtle);
    background: rgba(255,255,255,0.015);
}
.day-name {
    font-family: var(--font-display);
    font-size: 14px; font-weight: 700;
    color: var(--text-primary); letter-spacing: -0.01em;
}
.day-focus {
    font-size: 11px; font-weight: 500;
    color: var(--accent);
    background: var(--accent-subtle);
    border: 1px solid var(--accent-muted);
    padding: 3px 9px; border-radius: 999px;
}
.day-body { padding: 4px 0; }
.exercise-row {
    display: flex; align-items: center; justify-content: space-between;
    padding: 11px 18px;
    border-bottom: 1px solid var(--border-subtle);
    transition: background 150ms;
}
.exercise-row:hover { background: rgba(255,255,255,0.015); }
.exercise-row:last-child { border-bottom: none; }
.exercise-name {
    color: var(--text-primary);
    font-size: 13px; font-weight: 500;
    flex: 1; padding-right: 12px;
}
.exercise-meta { display: flex; align-items: center; gap: 12px; flex-shrink: 0; }
.exercise-detail {
    color: var(--text-faint);
    font-family: var(--font-mono);
    font-size: 11px; font-weight: 400;
}
.yt-btn {
    font-size: 10px;
    color: var(--accent);
    background: var(--accent-subtle);
    border: 1px solid var(--accent-muted);
    border-radius: 6px;
    padding: 4px 8px;
    text-decoration: none;
    font-weight: 500;
    letter-spacing: 0.02em;
    white-space: nowrap;
    transition: all 150ms;
}
.yt-btn:hover {
    background: var(--accent-muted);
    color: var(--accent-hover);
}

.rest-day-card {
    background: rgba(255,255,255,0.015);
    border: 1px dashed var(--border-default);
    border-radius: 12px;
    padding: 28px 18px;
    text-align: center;
    display: flex; flex-direction: column; justify-content: center;
    min-height: 140px;
}
.rest-label {
    font-family: var(--font-display);
    font-size: 13px; font-weight: 600;
    color: var(--text-faint); letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.rest-sub {
    font-size: 12px;
    color: var(--text-disabled);
}

/* ===== MACRO BAR ===== */
.macro-bar {
    background: var(--bg-surface);
    border: 1px solid var(--border-default);
    border-radius: 12px;
    padding: 22px 24px;
    margin-bottom: 14px;
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 0;
    box-shadow: var(--shadow-card);
}
.macro-cell {
    padding: 0 18px;
    border-right: 1px solid var(--border-subtle);
}
.macro-cell:first-child { padding-left: 0; }
.macro-cell:last-child { border-right: none; padding-right: 0; }
.macro-num {
    font-family: var(--font-mono);
    font-size: 28px; font-weight: 600;
    letter-spacing: -0.02em;
    color: var(--text-primary);
    line-height: 1;
}
.macro-cell.calories .macro-num { color: var(--accent); }
.macro-cell.protein  .macro-num { color: var(--ai-hover); }
.macro-sub {
    font-size: 11px; color: var(--text-faint);
    margin-top: 6px; font-weight: 400;
}
.macro-lbl {
    font-size: 10px; letter-spacing: 0.1em;
    text-transform: uppercase; color: var(--text-disabled);
    margin-top: 4px; font-weight: 600;
}
.macro-source {
    font-size: 11px; color: var(--text-disabled);
    margin-bottom: 28px;
}

/* ===== MEAL CARDS ===== */
.meal-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
    margin-bottom: 12px;
}
.meal-card {
    background: var(--bg-surface);
    border: 1px solid var(--border-default);
    border-radius: 10px;
    overflow: hidden;
    transition: all 200ms var(--ease-spring);
}
.meal-card:hover {
    border-color: var(--border-strong);
    transform: translateY(-1px);
}
.meal-card-header {
    padding: 10px 14px;
    background: var(--bg-elevated);
    border-bottom: 1px solid var(--border-subtle);
}
.meal-card-title {
    font-size: 10px; font-weight: 600;
    letter-spacing: 0.1em; text-transform: uppercase;
    color: var(--accent);
}
.meal-card-body {
    padding: 12px 14px;
    font-size: 12px;
    color: var(--text-secondary);
    line-height: 1.6;
}

/* ===== TRACKING TIPS ===== */
.tracking-grid {
    background: var(--bg-surface);
    border: 1px solid var(--border-default);
    border-radius: 12px;
    padding: 22px 24px;
    margin-bottom: 28px;
}
.tracking-title {
    font-size: 10px; letter-spacing: 0.12em; text-transform: uppercase;
    color: var(--text-faint); font-weight: 600; margin-bottom: 16px;
}
.tracking-items {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px 24px;
}
.tracking-item {
    font-size: 13px;
    color: var(--text-secondary);
    line-height: 1.6;
}
.tracking-item strong { color: var(--text-primary); font-weight: 600; }

/* ===== SUPPLEMENTS ===== */
.supps-card {
    background: var(--bg-surface);
    border: 1px solid var(--border-default);
    border-radius: 12px;
    padding: 22px 24px;
    margin-bottom: 28px;
}
.supps-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-top: 8px;
}
.supp-item {
    font-size: 12px;
    color: var(--text-secondary);
    line-height: 1.55;
    padding: 14px;
    background: var(--bg-elevated);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
}
.supp-name {
    font-family: var(--font-display);
    font-weight: 600;
    color: var(--text-primary);
    display: block;
    font-size: 13px;
    margin-bottom: 4px;
}
.supp-dose {
    font-size: 10px;
    color: var(--accent);
    font-family: var(--font-mono);
    font-weight: 600;
    display: block;
    margin-bottom: 8px;
    letter-spacing: 0.02em;
}

/* ===== TIPS ===== */
.tips-section {
    margin-top: 12px;
    background: linear-gradient(135deg, var(--bg-surface) 0%, var(--bg-elevated) 100%);
    border: 1px solid var(--border-default);
    border-radius: 14px;
    padding: 28px 32px;
    box-shadow: var(--shadow-card);
}
.tips-title {
    font-family: var(--font-display);
    font-size: 12px; letter-spacing: 0.12em; text-transform: uppercase;
    color: var(--accent); font-weight: 700; margin-bottom: 20px;
}
.tip-item {
    display: flex; gap: 16px;
    margin-bottom: 16px;
    align-items: flex-start;
    padding-bottom: 16px;
    border-bottom: 1px solid var(--border-subtle);
}
.tip-item:last-child { border-bottom: none; padding-bottom: 0; margin-bottom: 0; }
.tip-num {
    font-family: var(--font-display);
    font-size: 28px; font-weight: 800;
    color: var(--accent); line-height: 1;
    flex-shrink: 0; min-width: 32px;
    letter-spacing: -0.04em;
    background: linear-gradient(135deg, var(--accent) 0%, #C2410C 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
}
.tip-text {
    font-size: 13px;
    color: var(--text-secondary);
    line-height: 1.65;
    padding-top: 4px;
}

/* ===== TABS ===== */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid var(--border-default) !important;
    border-radius: 0 !important;
    padding: 0 48px !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 0 !important;
    font-family: var(--font-body) !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    color: var(--text-faint) !important;
    padding: 16px 24px 16px 0 !important;
    background: transparent !important;
    border: none !important;
    transition: color 200ms var(--ease-spring) !important;
}
.stTabs [data-baseweb="tab"]:hover { color: var(--text-secondary) !important; }
.stTabs [aria-selected="true"] {
    color: var(--text-primary) !important;
    border-bottom: 2px solid var(--accent) !important;
}
.stTabs [data-baseweb="tab-panel"] { padding: 0 !important; }

/* ===== CHAT ===== */
.chat-wrap { padding: 40px 48px; max-width: 1100px; margin: 0 auto; }
.chat-header {
    margin-bottom: 28px;
    padding-bottom: 20px;
    border-bottom: 1px solid var(--border-default);
}
.chat-title {
    font-family: var(--font-display);
    font-size: 32px;
    color: var(--text-primary);
    font-weight: 700; letter-spacing: -0.025em;
    margin-bottom: 8px;
    display: flex; align-items: center; gap: 12px;
}
.chat-badge {
    font-size: 10px; font-weight: 600;
    letter-spacing: 0.1em; text-transform: uppercase;
    background: var(--ai-muted); color: var(--ai-hover);
    border: 1px solid var(--ai); border-radius: 999px;
    padding: 4px 10px;
}
.chat-sub {
    font-size: 14px;
    color: var(--text-muted);
    line-height: 1.55;
    max-width: 720px;
}

.msg-you {
    background: var(--bg-surface);
    border: 1px solid var(--border-default);
    border-radius: 14px 14px 4px 14px;
    margin: 16px 0 16px 80px;
    overflow: hidden;
    box-shadow: var(--shadow-card);
}
.msg-you-label {
    font-size: 10px; letter-spacing: 0.1em; text-transform: uppercase;
    color: var(--text-faint); font-weight: 600;
    padding: 11px 16px 9px;
    border-bottom: 1px solid var(--border-subtle);
    text-align: right;
    font-family: var(--font-display);
}
.msg-you-body {
    font-size: 14px; color: var(--text-primary);
    line-height: 1.6; padding: 14px 18px;
}
.msg-ai {
    background: var(--bg-surface);
    border: 1px solid var(--ai-muted);
    border-left: 3px solid var(--ai);
    border-radius: 14px 14px 14px 4px;
    margin: 16px 80px 16px 0;
    overflow: hidden;
    box-shadow: var(--shadow-card);
}
.msg-ai-label {
    font-size: 10px; letter-spacing: 0.1em; text-transform: uppercase;
    color: var(--ai-hover); font-weight: 600;
    padding: 11px 16px 9px;
    border-bottom: 1px solid var(--border-subtle);
    font-family: var(--font-display);
    display: flex; align-items: center; gap: 8px;
}
.msg-ai-label::before {
    content: ''; width: 6px; height: 6px;
    background: var(--ai); border-radius: 50%;
    box-shadow: 0 0 8px var(--ai);
}
.msg-ai-body {
    font-size: 14px; color: var(--text-secondary);
    line-height: 1.8; padding: 16px 18px;
}
.msg-ai-body ol, .msg-ai-body ul { padding-left: 22px; margin: 8px 0; }
.msg-ai-body li { margin-bottom: 6px; line-height: 1.65; }
.msg-ai-body strong { color: var(--text-primary); font-weight: 600; }

.yt-link {
    display: inline-flex; align-items: center; gap: 6px;
    background: var(--accent-subtle);
    border: 1px solid var(--accent-muted);
    border-radius: 6px;
    padding: 6px 11px;
    font-size: 11px;
    color: var(--accent);
    font-weight: 500;
    text-decoration: none;
    margin: 6px 4px 0 0;
    transition: all 150ms;
}
.yt-link:hover {
    background: var(--accent-muted);
    color: var(--accent-hover);
}

.empty-state {
    text-align: center;
    padding: 80px 40px;
    border: 1px dashed var(--border-default);
    border-radius: 14px;
    margin: 20px 0;
    background: rgba(255,255,255,0.015);
}
.empty-icon {
    width: 56px; height: 56px;
    margin: 0 auto 16px;
    border-radius: 14px;
    background: var(--bg-surface);
    border: 1px solid var(--border-default);
    display: flex; align-items: center; justify-content: center;
    font-family: var(--font-display);
    font-size: 24px; font-weight: 800;
    color: var(--text-disabled);
}
.empty-title {
    font-family: var(--font-display);
    font-size: 22px;
    color: var(--text-muted);
    margin-bottom: 8px;
    font-weight: 600;
}
.empty-body {
    font-size: 14px;
    color: var(--text-faint);
    max-width: 420px;
    margin: 0 auto;
    line-height: 1.55;
}

/* ===== PROGRESS ===== */
.stProgress > div > div > div { background: var(--accent) !important; }
.stProgress > div > div { background: var(--bg-elevated) !important; }

/* ===== MOBILE ===== */
@media (max-width: 900px) {
    .hero { padding: 40px 20px 24px; }
    .hero-title { font-size: 36px; }
    .form-wrap { padding: 24px 20px; }
    .nav { padding: 14px 20px; }
    .plan-section { padding: 28px 20px; }
    .plan-grid { grid-template-columns: 1fr; }
    .stats-strip, .macro-bar { grid-template-columns: 1fr 1fr; gap: 16px; }
    .stat-cell, .macro-cell { border-right: none; padding: 0; }
    .meal-grid { grid-template-columns: 1fr 1fr; }
    .supps-grid { grid-template-columns: 1fr 1fr; }
    .tracking-items { grid-template-columns: 1fr; }
    .msg-you { margin-left: 20px; }
    .msg-ai { margin-right: 20px; }
    .chat-wrap { padding: 24px 20px; }
    .stTabs [data-baseweb="tab-list"] { padding: 0 20px !important; }
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE
# ============================================================
if "plan_data" not in st.session_state:
    st.session_state.plan_data = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "user_profile" not in st.session_state:
    st.session_state.user_profile = None

if "discipline" not in st.session_state:
    st.session_state.discipline = None

# ===========================
# NUTRITION CALCULATOR
# Jeff Nippard method — Pure Bodybuilding Nutrition Booklet
# ===========================
def calculate_nutrition(weight_kg, goal):
    weight_lbs = weight_kg * 2.205
    maintenance = int(weight_lbs * 16)  # bw x 16 as baseline

    if goal == "Bodybuilding":
        calories = int(maintenance * 1.10)      # 10% surplus (5-15% range)
        protein_g = int(weight_kg * 2.0)        # 1.6-2.2g/kg for bulking
    elif goal == "Lose Weight":
        calories = int(maintenance * 0.90)      # 10% deficit (5-15% range)
        protein_g = int(weight_kg * 2.2)        # higher protein in deficit
    elif goal == "Body Recomposition":
        calories = maintenance                   # maintenance calories
        protein_g = int(weight_kg * 2.5)        # very high protein for recomp
    elif goal == "Improve Endurance":
        calories = int(maintenance * 1.05)
        protein_g = int(weight_kg * 1.7)        # 1.6-1.8g/kg
    else:  # Maintain
        calories = maintenance
        protein_g = int(weight_kg * 1.8)

    fat_g = int((calories * 0.25) / 9)         # 25% of cals from fat (20-30% range)
    protein_cals = protein_g * 4
    fat_cals = fat_g * 9
    carb_g = max(0, int((calories - protein_cals - fat_cals) / 4))  # fills remainder

    return {
        "calories": calories,
        "maintenance": maintenance,
        "protein_g": protein_g,
        "fat_g": fat_g,
        "carb_g": carb_g
    }

# ===========================
# YOUTUBE LINKS DATABASE
# ===========================
YOUTUBE_LINKS = {
    "squat": ("How to Squat — Joab Murphy", "https://www.youtube.com/watch?v=C8MYeGhmn-U"),
    "deadlift": ("How to Deadlift — Joab Murphy", "https://www.youtube.com/watch?v=ytGaGIn3SjE"),
    "bench press": ("How to Bench Press — Joab Murphy", "https://www.youtube.com/watch?v=gRVjAtPip0Y"),
    "overhead press": ("How to Overhead Press", "https://www.youtube.com/watch?v=2yjwXTZQDDI"),
    "pull up": ("Pull Ups — Joab Murphy", "https://www.youtube.com/watch?v=IpxAG2z91Ys"),
    "lat pulldown": ("Lat Pulldown — Joab Murphy", "https://www.youtube.com/watch?v=0oeIB6wi3es"),
    "row": ("Barbell Row — Joab Murphy", "https://www.youtube.com/watch?v=vT2GjY_Umpw"),
    "good morning": ("Good Mornings", "https://www.youtube.com/watch?v=vKPGe8zb2S4"),
    "leg press": ("Leg Press", "https://www.youtube.com/watch?v=IZxyjW7MPJQ"),
    "leg extension": ("Leg Extensions", "https://www.youtube.com/watch?v=YyvSfVjQeL0"),
    "leg curl": ("Leg Curl — Joab Murphy", "https://www.youtube.com/watch?v=GbcgewbuwMI"),
    "lateral raise": ("Lateral Raise — Jeff Nippard", "https://www.youtube.com/watch?v=zpUTA5i16kA"),
    "bicep": ("Bicep Curl — Joab Murphy", "https://www.youtube.com/watch?v=QZEqB6wUPxQ"),
    "tricep": ("Tricep Extension", "https://www.youtube.com/watch?v=SLYwsE_W1eM"),
    "skull crusher": ("Skull Crushers", "https://www.youtube.com/watch?v=3rrrpsRKDi0"),
    "plank": ("Plank Form", "https://www.youtube.com/watch?v=pSHjTRCQxIw"),
    "dead bug": ("Dead Bugs", "https://www.youtube.com/watch?v=g_BYB0R-4Ws"),
    "side plank": ("Side Plank", "https://www.youtube.com/watch?v=N_s9em1xTqU"),
    "romanian deadlift": ("Romanian Deadlift — Jeff Nippard", "https://www.youtube.com/watch?v=2SHsk9AzdjA"),
    "hip thrust": ("Hip Thrust — Bret Contreras", "https://www.youtube.com/watch?v=xDmFkJxPzeM"),
    "progressive overload": ("Progressive Overload — Jeff Nippard", "https://www.youtube.com/watch?v=GEMkFNQYPuI"),
    "protein": ("How Much Protein — Jeff Nippard", "https://www.youtube.com/watch?v=GPC5HMJ30ZM"),
    "creatine": ("Creatine Guide — Jeff Nippard", "https://www.youtube.com/watch?v=9FkdR6Zyf2E"),
    "calorie": ("Calorie Deficit — Jeff Nippard", "https://www.youtube.com/watch?v=EISZ9T7RmN0"),
    "recomposition": ("Body Recomp Guide — Jeff Nippard", "https://www.youtube.com/watch?v=b1v2L0Kcxpw"),
    "hiit": ("HIIT Guide — Jeremy Ethier", "https://www.youtube.com/watch?v=fBFT8F_gLPw"),
    "goblet squat": ("Goblet Squat", "https://www.youtube.com/watch?v=gpNky6gvseQ"),
    "back extension": ("Back Extensions — Joab Murphy", "https://www.youtube.com/watch?v=ph3pddpKzzw"),
    "safety bar": ("Safety Bar Squat", "https://www.youtube.com/watch?v=b2jmZyptN64"),
    "push up": ("Elevated Push Up", "https://www.youtube.com/watch?v=KadL9HpmWSg"),
}

def get_youtube_links(message):
    message_lower = message.lower()
    found = []
    for keyword, (title, url) in YOUTUBE_LINKS.items():
        if keyword in message_lower and len(found) < 3:
            found.append((title, url))
    return found

# ===========================
# DISCIPLINE DATA
# Sources: Joab Murphy MSc BSc (powerlifting), Jeff Nippard (bodybuilding + nutrition)
# ===========================
DISCIPLINES = {

    "Bodybuilding": {
        "note": "Bodybuilding is about building muscle size through volume, progressive overload, and the mind-muscle connection. Every rep should be felt in the target muscle — not just moved.",
        "source": "Based on Jeff Nippard — Pure Bodybuilding Program Phase 2 (Upper/Lower Split)",
        "focus_days": {
            # Jeff Nippard Upper/Lower split
            1: ("Upper — Strength Focus", [
                ("Barbell Bench Press", "3-4×3-5 heavy", "2-3 min", "https://www.youtube.com/watch?v=gRVjAtPip0Y"),
                ("Barbell Row", "3-4×3-5 heavy", "2-3 min", "https://www.youtube.com/watch?v=vT2GjY_Umpw"),
                ("Incline Dumbbell Press", "3×8-12", "90s", ""),
                ("Cable Row", "3×10-12", "75s", ""),
                ("Lateral Raise", "3×15-20", "60s", "https://www.youtube.com/watch?v=zpUTA5i16kA"),
                ("Bicep Curl", "2×10-15", "60s", "https://www.youtube.com/watch?v=QZEqB6wUPxQ"),
            ]),
            2: ("Lower — Strength Focus", [
                ("Back Squat", "3-4×3-5 heavy", "2-3 min", "https://www.youtube.com/watch?v=C8MYeGhmn-U"),
                ("Romanian Deadlift", "3×8-10", "2 min", "https://www.youtube.com/watch?v=2SHsk9AzdjA"),
                ("Leg Press", "3×10-12", "90s", "https://www.youtube.com/watch?v=IZxyjW7MPJQ"),
                ("Leg Curl", "3×12-15", "60s", "https://www.youtube.com/watch?v=GbcgewbuwMI"),
                ("Calf Raise", "4×15-20", "45s", ""),
            ]),
            3: ("Upper — Hypertrophy Focus", [
                ("Incline Bench Press", "3×8-12", "90s", ""),
                ("Lat Pulldown", "3×10-12", "75s", "https://www.youtube.com/watch?v=0oeIB6wi3es"),
                ("Pec Deck or Cable Fly", "3×12-15", "60s", ""),
                ("Face Pull", "3×15-20", "60s", ""),
                ("Tricep Pushdown", "3×12-15", "60s", "https://www.youtube.com/watch?v=SLYwsE_W1eM"),
                ("Hammer Curl", "3×12-15", "60s", ""),
            ]),
            4: ("Lower — Hypertrophy Focus", [
                ("Hack Squat or Leg Press", "3×10-15", "90s", ""),
                ("Bulgarian Split Squat", "3×10-12", "90s", ""),
                ("Leg Extension", "3×15-20", "60s", "https://www.youtube.com/watch?v=YyvSfVjQeL0"),
                ("Seated Leg Curl", "3×15-20", "60s", ""),
                ("Calf Raise", "4×20-25", "45s", ""),
            ]),
            5: ("Upper — Volume Day", [
                ("Dumbbell Overhead Press", "3×10-12", "75s", "https://www.youtube.com/watch?v=2yjwXTZQDDI"),
                ("Pull-ups", "3×8-12", "90s", "https://www.youtube.com/watch?v=IpxAG2z91Ys"),
                ("Cable Fly", "3×15", "60s", ""),
                ("Lateral Raise", "3×20", "45s", "https://www.youtube.com/watch?v=zpUTA5i16kA"),
                ("Skull Crushers", "3×12", "60s", "https://www.youtube.com/watch?v=3rrrpsRKDi0"),
                ("Preacher Curl", "3×12", "60s", ""),
            ]),
            6: ("Lower — Volume Day", [
                ("Front Squat or Goblet Squat", "3×10-12", "90s", ""),
                ("Walking Lunge", "3×12 each leg", "75s", ""),
                ("Leg Extension", "3×20", "60s", ""),
                ("Leg Curl", "3×20", "60s", ""),
                ("Calf Raise", "4×25", "45s", ""),
            ]),
        },
        "meals": {
            "Breakfast": "Oats + whey protein shake + banana + almond butter",
            "Lunch": "Chicken breast + brown rice + salad + olive oil dressing",
            "Pre-workout": "Greek yoghurt + berries + honey (1 hr before)",
            "Dinner": "Salmon + sweet potato + green beans + butter"
        },
        "tips": [
            "Control the eccentric (lowering) phase — take 2-3 seconds to lower the weight. The muscle damage from this phase drives more growth than the lifting phase.",
            "Train each muscle group twice per week for optimal hypertrophy (Jeff Nippard). This Upper/Lower split achieves this automatically — chest on Day 1 and Day 3.",
            "Progressive overload in bodybuilding: add reps first, then weight. Once you hit the top of your rep range on all sets, add 2.5-5kg and start again.",
            "Track your measurements monthly — chest, arms, legs, waist. The scale is misleading. Measurements and photos tell you if the program is actually working."
        ]
    },

    "Body Recomposition": {
        "note": "Body recomposition — losing fat and building muscle simultaneously — requires maintenance calories, very high protein, and a combination of heavy compound work with cardio.",
        "source": "Based on Jeff Nippard recomposition principles — Pure Bodybuilding Nutrition Booklet",
        "focus_days": {
            1: ("Upper Body Strength", [
                ("Bench Press", "4×6-8", "2 min", "https://www.youtube.com/watch?v=gRVjAtPip0Y"),
                ("Barbell Row", "4×6-8", "2 min", "https://www.youtube.com/watch?v=vT2GjY_Umpw"),
                ("Overhead Press", "3×8-10", "90s", "https://www.youtube.com/watch?v=2yjwXTZQDDI"),
                ("Pull-ups", "3×8-10", "90s", "https://www.youtube.com/watch?v=IpxAG2z91Ys"),
            ]),
            2: ("LISS Cardio + Core", [
                ("Treadmill walk — incline 10%", "40 min", "—", ""),
                ("Side Plank", "3×30s each side", "—", "https://www.youtube.com/watch?v=N_s9em1xTqU"),
                ("Dead Bugs", "3×10", "—", "https://www.youtube.com/watch?v=g_BYB0R-4Ws"),
                ("Plank", "3×45s", "—", ""),
            ]),
            3: ("Lower Body Strength", [
                ("Back Squat", "4×6-8", "2 min", "https://www.youtube.com/watch?v=C8MYeGhmn-U"),
                ("Romanian Deadlift", "3×10", "90s", "https://www.youtube.com/watch?v=2SHsk9AzdjA"),
                ("Leg Press", "3×12", "90s", "https://www.youtube.com/watch?v=IZxyjW7MPJQ"),
                ("Walking Lunge", "3×12 each leg", "75s", ""),
            ]),
            4: ("HIIT Cardio + Core", [
                ("Sprint intervals — 30s on / 30s off", "10 rounds", "—", "https://www.youtube.com/watch?v=fBFT8F_gLPw"),
                ("Dead Bugs", "3×10", "—", "https://www.youtube.com/watch?v=g_BYB0R-4Ws"),
                ("Plank", "3×60s", "—", ""),
            ]),
            5: ("Full Body Compound", [
                ("Deadlift", "4×5", "2 min", "https://www.youtube.com/watch?v=ytGaGIn3SjE"),
                ("Incline Bench Press", "3×10", "90s", ""),
                ("Lat Pulldown", "3×12", "75s", "https://www.youtube.com/watch?v=0oeIB6wi3es"),
                ("Bulgarian Split Squat", "3×10", "90s", ""),
            ]),
            6: ("Active Recovery", [
                ("Light cycling or walking", "30 min", "—", ""),
                ("Foam rolling — full body", "15 min", "—", ""),
                ("Stretching", "15 min", "—", ""),
            ]),
        },
        "meals": {
            "Breakfast": "Egg white omelette (4 whites + 2 whole) + oats + black coffee",
            "Lunch": "Turkey breast + quinoa + roasted vegetables",
            "Pre-workout": "Banana + whey protein shake (fuels training day)",
            "Dinner": "White fish + sweet potato + asparagus + lemon"
        },
        "tips": [
            "Protein is the single most important variable for recomposition. At 2.5g per kg of bodyweight, your body has everything it needs to build muscle even in a slight deficit.",
            "Carb cycle — eat more carbs on training days (around workouts) and fewer carbs on rest days. Training day: slight surplus. Rest day: slight deficit.",
            "Do not chase the scale. Recomposition is slow — the scale may barely move for months. Monthly photos and measurements are the only reliable progress metric.",
            "Prioritise sleep. Growth hormone is released during deep sleep — this is when fat is mobilised and muscle is repaired and built. 8 hours is the target."
        ]
    },

    "Lose Weight": {
        "note": "Fat loss requires a consistent caloric deficit combined with resistance training to preserve muscle. Cardio accelerates the process and improves cardiovascular health.",
        "source": "Based on Jeff Nippard fat loss principles — 5-15% caloric deficit recommended",
        "focus_days": {
            1: ("Full Body Resistance", [
                ("Goblet Squat", "3×15", "60s", "https://www.youtube.com/watch?v=gpNky6gvseQ"),
                ("Push-ups or Bench Press", "3×15", "60s", ""),
                ("Dumbbell Row", "3×15", "60s", "https://www.youtube.com/watch?v=vT2GjY_Umpw"),
                ("Hip Thrust", "3×15", "60s", "https://www.youtube.com/watch?v=xDmFkJxPzeM"),
            ]),
            2: ("HIIT Cardio", [
                ("Jump rope or sprints — 30s on / 30s off", "15 min", "—", "https://www.youtube.com/watch?v=fBFT8F_gLPw"),
                ("Burpees", "3×15", "60s", ""),
                ("Mountain Climbers", "3×30s", "—", ""),
                ("Plank", "3×45s", "—", ""),
            ]),
            3: ("Upper Body", [
                ("Dumbbell Press", "3×15", "60s", ""),
                ("Lat Pulldown", "3×15", "60s", "https://www.youtube.com/watch?v=0oeIB6wi3es"),
                ("Lateral Raise", "3×20", "45s", "https://www.youtube.com/watch?v=zpUTA5i16kA"),
                ("Tricep Pushdown", "3×20", "45s", ""),
            ]),
            4: ("Steady State Cardio", [
                ("Brisk walk or jog", "45 min", "—", ""),
                ("Incline treadmill walk if needed", "20 min", "—", ""),
            ]),
            5: ("Lower Body", [
                ("Leg Press", "3×15", "60s", "https://www.youtube.com/watch?v=IZxyjW7MPJQ"),
                ("Walking Lunge", "3×12", "60s", ""),
                ("Leg Curl", "3×15", "60s", ""),
                ("Calf Raise", "3×25", "45s", ""),
            ]),
            6: ("Active Recovery", [
                ("Light walk", "30 min", "—", ""),
                ("Yoga or full body stretching", "20 min", "—", ""),
            ]),
        },
        "meals": {
            "Breakfast": "Greek yoghurt + berries + protein powder — high protein, low calorie start",
            "Lunch": "Large salad + grilled chicken + olive oil + lemon + boiled egg",
            "Snack": "Apple + 20g almonds + protein shake if needed",
            "Dinner": "Steamed white fish + large portion vegetables + small rice portion"
        },
        "tips": [
            "Track everything you eat for the first 4 weeks. Most people underestimate their intake by 20-30%. Use MyFitnessPal or Cronometer and weigh your food with a scale.",
            "A 10% caloric deficit is sustainable. A 25% deficit feels faster but causes muscle loss, hunger, and rebound weight gain. Slow and steady wins every time.",
            "Do not cut carbs entirely — reduce them. Carbs fuel your resistance training sessions. Without them, training quality drops and you burn less overall calories.",
            "Weigh yourself daily and take the weekly average. Daily weight fluctuates 1-3kg due to water, food, and hormones. The 7-day average shows real fat loss progress."
        ]
    },

    "Improve Endurance": {
        "note": "Endurance training builds your aerobic base, increases stamina, and improves cardiovascular health. Consistency and gradual progression matter more than intensity.",
        "source": "Based on Norwegian 4×4 intervals, Zone 2 training, and periodization principles",
        "focus_days": {
            1: ("Zone 2 Base Run", [
                ("Easy jog — conversational pace", "30-40 min", "—", ""),
                ("Heart rate target: 60-70% of max HR", "—", "—", ""),
            ]),
            2: ("Strength Support", [
                ("Goblet Squat", "3×12", "90s", "https://www.youtube.com/watch?v=gpNky6gvseQ"),
                ("Deadlift", "3×8", "2 min", "https://www.youtube.com/watch?v=ytGaGIn3SjE"),
                ("Calf Raise", "3×20", "60s", ""),
                ("Plank + Dead Bug", "3×30s", "—", ""),
            ]),
            3: ("Tempo Run", [
                ("Warm up jog", "10 min", "—", ""),
                ("Comfortably hard pace — 20 min sustained", "20 min", "—", ""),
                ("Cool down walk", "10 min", "—", ""),
            ]),
            4: ("Cross Training", [
                ("Cycling or swimming — steady pace", "40 min", "—", ""),
                ("Low impact and steady", "—", "—", ""),
            ]),
            5: ("HIIT Intervals — Norwegian 4×4", [
                ("4 min hard effort / 3 min easy", "4 rounds", "3 min", "https://www.youtube.com/watch?v=fBFT8F_gLPw"),
                ("Cool down jog", "10 min", "—", ""),
            ]),
            6: ("Long Slow Distance", [
                ("Long easy run — build distance weekly", "50-60 min", "—", ""),
                ("Heart rate: 60-65% max throughout", "—", "—", ""),
            ]),
        },
        "meals": {
            "Breakfast": "Porridge + banana + honey + whole milk — carb rich to fuel training",
            "Pre-run": "Toast + peanut butter + orange juice (1-2 hrs before)",
            "During long runs": "Energy gel or banana every 45 min for sessions over 60 min",
            "Recovery meal": "Pasta + lean mince + tomato sauce + parmesan within 30 min"
        },
        "tips": [
            "Build distance by no more than 10% per week. Most endurance injuries come from doing too much too soon. Patience in month 1 pays off in month 6.",
            "80% of your runs should feel easy — conversational pace, Zone 2. Most beginners run everything too hard. Easy days build your aerobic base.",
            "Carbohydrates are not optional — they are your fuel. On long training days you need 6-8g of carbs per kg of bodyweight. Do not restrict them.",
            "Strength train twice a week. Strong legs, glutes, and core prevent injury and improve running economy significantly. Do not skip strength sessions."
        ]
    },

    "Maintain Fitness": {
        "note": "Maintenance is about sustainability. The best program is the one you can stick to for years. Balance and consistency beat short-term intensity every single time.",
        "source": "Based on general fitness principles and auto-regulated training",
        "focus_days": {
            1: ("Full Body Strength", [
                ("Squat", "3×10", "90s", "https://www.youtube.com/watch?v=C8MYeGhmn-U"),
                ("Bench Press", "3×10", "90s", "https://www.youtube.com/watch?v=gRVjAtPip0Y"),
                ("Barbell Row", "3×10", "90s", "https://www.youtube.com/watch?v=vT2GjY_Umpw"),
                ("Overhead Press", "3×10", "90s", "https://www.youtube.com/watch?v=2yjwXTZQDDI"),
            ]),
            2: ("Cardio + Mobility", [
                ("Brisk walk or cycle", "30 min", "—", ""),
                ("Full body stretch", "15 min", "—", ""),
            ]),
            3: ("Full Body Strength", [
                ("Deadlift", "3×8", "2 min", "https://www.youtube.com/watch?v=ytGaGIn3SjE"),
                ("Dumbbell Press", "3×12", "75s", ""),
                ("Pull-ups or Lat Pulldown", "3×12", "75s", "https://www.youtube.com/watch?v=0oeIB6wi3es"),
                ("Lunges", "3×12", "60s", ""),
            ]),
            4: ("Active Recovery", [
                ("Yoga or Pilates", "30 min", "—", ""),
                ("Light walk", "20 min", "—", ""),
            ]),
            5: ("Full Body + Core", [
                ("Goblet Squat", "3×12", "75s", ""),
                ("Push-ups", "3×20", "60s", ""),
                ("Cable Row", "3×12", "75s", ""),
                ("Plank", "3×60s", "—", ""),
            ]),
            6: ("Favourite Cardio", [
                ("Whatever you enjoy — run, swim, cycle, sport", "30-45 min", "—", ""),
            ]),
        },
        "meals": {
            "Breakfast": "2 eggs + whole grain toast + avocado + coffee",
            "Lunch": "Chicken wrap + salad + fruit — balanced and practical",
            "Snack": "Nuts + yoghurt or protein bar if needed",
            "Dinner": "Protein + vegetable + carb of your choice — enjoy your food"
        },
        "tips": [
            "Consistency over perfection. Showing up 3 days every week for 5 years beats 6 days a week for 3 months then quitting. The habit is worth more than the program.",
            "Find movement you actually enjoy. If you hate running, stop. Walk, swim, cycle, play a sport. Activity you enjoy is activity you will sustain.",
            "Maintenance calories are not a restriction. You are fuelling an active lifestyle. Eat enough protein and do not feel guilty about your food choices.",
            "Even at maintenance, chase small improvements. Add a rep, add a kilo, improve your form. Progress does not require a major goal — it keeps training engaging."
        ]
    }
}

@st.cache_resource
def load_local_model():
    import os
    if not os.path.exists("gym_ai_model/adapter_config.json"):
        return None, None
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained("gym_ai_model")
    base_name = "mistralai/Mistral-7B-Instruct-v0.3"
    model = AutoModelForCausalLM.from_pretrained(base_name, quantization_config=bnb_config, device_map="auto")
    model = PeftModel.from_pretrained(model, "gym_ai_model")
    return model, tokenizer

def md_to_html(text):
    import re
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)
    lines = text.split('\n')
    out = []
    for line in lines:
        s = line.strip()
        if re.match(r'^[-•]\s+', s):
            s = re.sub(r'^[-•]\s+', '', s)
            out.append(f'<li>{s}</li>')
        elif re.match(r'^\d+\.\s+', s):
            s = re.sub(r'^\d+\.\s+', '', s)
            out.append(f'<li>{s}</li>')
        elif s:
            out.append(f'<p style="margin:4px 0">{s}</p>')
        else:
            out.append('<br>')
    return ''.join(out)

def generate_response_local(model, tokenizer, system_prompt, messages):
    chat = [{"role": "system", "content": system_prompt}] + messages
    text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    encoded = tokenizer(text, return_tensors="pt").to("cuda")
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 2
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=1024,
            temperature=0.7,
            do_sample=True,
            pad_token_id=pad_id,
            repetition_penalty=1.1
        )
    return tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True).strip()

# ============================================================
# NAV
# ============================================================
st.markdown("""
<div class="nav">
    <div class="nav-logo">
        <div class="nav-mark">F</div>
        FitBy<span>Minh</span>
    </div>
    <div class="nav-status">
        <div class="status-dot"></div>
        AI Trainer Online
    </div>
</div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Build my plan", "Talk to my trainer"])

# ============================================================
# TAB 1 — BUILD PLAN
# ============================================================
with tab1:
    st.markdown("""
    <div class="hero">
        <div class="hero-eyebrow">Discipline-specific · Science-backed</div>
        <h1 class="hero-title">A plan built around <em>how</em> you actually train.</h1>
        <p class="hero-body">Choose your discipline, share your stats, and get a structured weekly schedule with real exercises, personalized macros, and meal plans — calculated for your specific body and goals.</p>
    </div>
    """, unsafe_allow_html=True)

    # ABOUT YOU
    st.markdown('<div class="form-wrap"><div class="section-label">About you</div></div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1: gender = st.selectbox("Gender", ["Male", "Female"])
    with c2: age = st.number_input("Age", 16, 80, 22)
    with c3: weight = st.number_input("Weight (kg)", 40, 150, 75)
    with c4: height = st.number_input("Height (cm)", 140, 220, 178)

    bmi = round(weight / ((height / 100) ** 2), 1)
    ideal = round(22 * ((height / 100) ** 2), 1)
    diff = round(weight - ideal, 1)
    diff_str = f"+{diff}kg" if diff > 0 else f"{diff}kg" if diff < 0 else "ideal"
    if bmi < 18.5: bmi_cat, bmi_cls = "Underweight", "warning"
    elif bmi < 25: bmi_cat, bmi_cls = "Healthy", "healthy"
    elif bmi < 30: bmi_cat, bmi_cls = "Overweight", "warning"
    else: bmi_cat, bmi_cls = "Obese", "danger"

    st.markdown(f"""
    <div class="form-wrap" style="padding-top:0;">
    <div class="stats-strip">
        <div class="stat-cell">
            <div class="stat-num accent">{bmi}</div>
            <div class="stat-lbl">BMI</div>
            <div class="stat-tag {bmi_cls}">{bmi_cat}</div>
        </div>
        <div class="stat-cell">
            <div class="stat-num">{ideal}<span style="font-size:14px;color:var(--text-faint);">kg</span></div>
            <div class="stat-lbl">Ideal weight</div>
        </div>
        <div class="stat-cell">
            <div class="stat-num">{diff_str}</div>
            <div class="stat-lbl">From ideal</div>
        </div>
        <div class="stat-cell">
            <div class="stat-num">{weight * 2.205:.0f}<span style="font-size:14px;color:var(--text-faint);">lbs</span></div>
            <div class="stat-lbl">In pounds</div>
        </div>
        <div class="stat-cell">
            <div class="stat-num">{age}<span style="font-size:14px;color:var(--text-faint);">yrs</span></div>
            <div class="stat-lbl">Age</div>
        </div>
    </div>
    </div>
    """, unsafe_allow_html=True)

    # DISCIPLINE
    st.markdown('<div class="form-wrap"><div class="section-label">Your discipline</div></div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        discipline = st.selectbox("Choose your training discipline", list(DISCIPLINES.keys()))
    with c2:
        experience = st.selectbox("Training experience", [
            "Just getting started", "A few months", "1-2 years", "3+ years"
        ])
    exp_map = {"Just getting started": "complete beginner", "A few months": "beginner", "1-2 years": "intermediate", "3+ years": "advanced"}

    # TRAINING SETUP
    st.markdown('<div class="form-wrap"><div class="section-label">Training setup</div></div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        equipment = st.selectbox("Equipment available", ["Full gym access", "Dumbbells only", "Barbell and dumbbells", "Bodyweight only", "Resistance bands only"])
    with c2:
        days = st.slider("Training days per week", 3, 6, 4)
        rec_map = {"Bodybuilding": 4, "Body Recomposition": 5, "Lose Weight": 5, "Improve Endurance": 5, "Maintain Fitness": 3}
        rec = rec_map.get(discipline, 4)
        if days == rec: st.caption("Optimal for this discipline")
        elif days < rec: st.caption(f"{rec} days recommended for {discipline}")
        else: st.caption("High volume — recovery is critical")
    with c3:
        gym_time = st.selectbox("Time per session", ["45 minutes", "60 minutes", "75 minutes", "90+ minutes"])
    with c4:
        diet = st.selectbox("Dietary preference", ["No restrictions", "Vegetarian", "Vegan", "Halal", "Gluten free", "Lactose intolerant"])

    time_config = {
        "45 minutes": {"max_ex": 3, "rest_mult": 0.6, "note": "45 min — top 3 exercises only, shorter rest. Ask your trainer to adjust."},
        "60 minutes": {"max_ex": 4, "rest_mult": 0.8, "note": "60 min — 4 main exercises, standard rest."},
        "75 minutes": {"max_ex": 5, "rest_mult": 1.0, "note": "75 min — full exercise selection, complete rest periods."},
        "90+ minutes": {"max_ex": 6, "rest_mult": 1.2, "note": "90+ min — full volume with all accessories and warm-up sets."}
    }
    t_cfg = time_config[gym_time]

    # TRAINER NOTE
    disc_data = DISCIPLINES[discipline]
    nutrition = calculate_nutrition(weight, discipline)

    st.markdown(f"""
    <div class="form-wrap" style="padding-top:0;">
    <div class="trainer-note">
        <div class="note-label">About {discipline}</div>
        <div class="note-main">{disc_data['note']}</div>
        <div class="note-sub">You will train {days} days per week as a {exp_map[experience]} with {gym_time} per session. Your estimated maintenance calories are {nutrition['maintenance']} kcal — your plan is calculated from this. {t_cfg['note']}</div>
    </div>
    </div>
    """, unsafe_allow_html=True)

    # GENERATE BUTTON
    st.markdown('<div class="form-wrap" style="padding-top:0; padding-bottom:48px;">', unsafe_allow_html=True)
    _, mid_col, _ = st.columns([3, 2, 3])
    with mid_col:
        generate = st.button("Generate my plan")
    st.markdown('</div>', unsafe_allow_html=True)

    if generate:
        user_input = f"Gender: {gender}, Age: {age}, Weight: {weight}kg, Height: {height}cm, BMI: {bmi}, Discipline: {discipline}, Experience: {exp_map[experience]}, Equipment: {equipment}, Diet: {diet}, Days: {days}, Time per session: {gym_time}"
        st.session_state.user_profile = user_input
        st.session_state.discipline = discipline
        st.session_state.chat_history = []

        all_days = list(disc_data["focus_days"].items())
        selected_days = all_days[:days]

        week_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        if days == 3: training_idx = [0, 2, 4]
        elif days == 4: training_idx = [0, 1, 3, 4]
        elif days == 5: training_idx = [0, 1, 2, 4, 5]
        else: training_idx = [0, 1, 2, 3, 4, 5]

        week_plan = {}
        t_idx = 0
        for i, day_name in enumerate(week_days):
            if i in training_idx and t_idx < len(selected_days):
                _, (focus, exercises) = selected_days[t_idx]
                week_plan[day_name] = {"type": "training", "focus": focus, "exercises": exercises}
                t_idx += 1
            else:
                week_plan[day_name] = {"type": "rest"}

        st.session_state.plan_data = {
            "week": week_plan,
            "nutrition": nutrition,
            "meals": disc_data["meals"],
            "tips": disc_data["tips"],
            "discipline": discipline,
            "source": disc_data["source"],
            "max_ex": t_cfg["max_ex"],
            "rest_mult": t_cfg["rest_mult"],
            "gym_time": gym_time,
            "time_note": t_cfg["note"]
        }

        steps = ["Reading your profile", f"Loading {discipline} program", "Building your weekly schedule", "Calculating personalized macros", "Writing your meal plan", "Done"]
        prog = st.progress(0)
        status = st.empty()
        for i, s in enumerate(steps):
            status.markdown(f"<p style='color:var(--text-faint);font-size:13px;padding:0 48px;font-family:var(--font-mono);'>&gt; {s}...</p>", unsafe_allow_html=True)
            prog.progress((i + 1) / len(steps))
            time.sleep(0.4)
        prog.empty()
        status.empty()

    # ============================================================
    # PLAN OUTPUT
    # ============================================================
    if st.session_state.plan_data:
        pd = st.session_state.plan_data
        nut = pd["nutrition"]

        def sanitize(text):
            return str(text).replace("×", "x").replace("&", "and").replace('"', "'")

        def calc_rest(ex_rest, mult):
            if ex_rest in ["—", "", None]:
                return "—"
            try:
                first_num = ex_rest.split("-")[0].strip()
                digits = ''.join(filter(str.isdigit, first_num.split()[0]))
                if not digits:
                    return ex_rest
                num = int(digits)
                if "min" in ex_rest:
                    total_s = num * 60
                else:
                    total_s = num
                adj = max(30, int(total_s * mult))
                if adj >= 60:
                    m, s = adj // 60, adj % 60
                    return f"{m}m {s}s rest" if s else f"{m} min rest"
                return f"{adj}s rest"
            except:
                return ex_rest

        st.markdown(f"""
        <div class="plan-section">
        <div class="plan-wrap">
            <div class="plan-header">
                <div class="plan-title-block">
                    <div class="plan-title">Your {pd['discipline']} Plan</div>
                    <div class="plan-meta">
                        <span class="meta-pill accent">{pd['gym_time']} per session</span>
                    </div>
                </div>
            </div>
        </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="adjust-banner">
            <div class="adjust-banner-icon">AI</div>
            <div>
                <strong>This plan is your starting point.</strong> Every body responds differently to training.
                Switch to <strong>Talk to my trainer</strong> to swap exercises, adjust rest periods,
                or ask anything about your program.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="subsection-label">Weekly training schedule</div>', unsafe_allow_html=True)

        # Weekly schedule — render each card individually
        st.markdown('<div class="plan-grid">', unsafe_allow_html=True)
        for day_name, day_data in pd["week"].items():
            if day_data["type"] == "training":
                exercises_to_show = day_data["exercises"][:pd["max_ex"]]
                rows = ""
                for ex_name, ex_sets, ex_rest, ex_yt in exercises_to_show:
                    adj_rest = calc_rest(ex_rest, pd["rest_mult"])
                    yt_btn = f'<a class="yt-btn" href="{ex_yt}" target="_blank">Watch</a>' if ex_yt else '<span style="font-size:10px;color:var(--text-disabled);font-family:var(--font-mono);">No video</span>'
                    rows += f'<div class="exercise-row"><span class="exercise-name">{sanitize(ex_name)}</span><div class="exercise-meta"><span class="exercise-detail">{sanitize(ex_sets)} · {adj_rest}</span>{yt_btn}</div></div>'
                st.markdown(f"""
                <div class="day-card">
                    <div class="day-header">
                        <span class="day-name">{sanitize(day_name)}</span>
                        <span class="day-focus">{sanitize(day_data['focus'])}</span>
                    </div>
                    <div class="day-body">{rows}</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="rest-day-card">
                    <div class="rest-label">{sanitize(day_name)} — Rest</div>
                    <div class="rest-sub">Light walking or stretching encouraged</div>
                </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # MACROS
        protein_per_kg = round(nut['protein_g']/weight, 1)
        st.markdown(f"""
        <div class="subsection-label">Personalized daily nutrition</div>
        <div class="macro-bar">
            <div class="macro-cell calories">
                <div class="macro-num">{nut['calories']}</div>
                <div class="macro-sub">kcal per day</div>
                <div class="macro-lbl">Calories</div>
            </div>
            <div class="macro-cell protein">
                <div class="macro-num">{nut['protein_g']}<span style="font-size:14px;">g</span></div>
                <div class="macro-sub">{protein_per_kg}g per kg bodyweight</div>
                <div class="macro-lbl">Protein</div>
            </div>
            <div class="macro-cell">
                <div class="macro-num">{nut['carb_g']}<span style="font-size:14px;">g</span></div>
                <div class="macro-sub">fills remaining calories</div>
                <div class="macro-lbl">Carbohydrates</div>
            </div>
            <div class="macro-cell">
                <div class="macro-num">{nut['fat_g']}<span style="font-size:14px;">g</span></div>
                <div class="macro-sub">25% of total calories</div>
                <div class="macro-lbl">Fats</div>
            </div>
            <div class="macro-cell">
                <div class="macro-num">{nut['maintenance']}</div>
                <div class="macro-sub">your baseline (bw × 16)</div>
                <div class="macro-lbl">Maintenance</div>
            </div>
        </div>
        <div class="macro-source">Adjust based on weekly weight trends.</div>
        """, unsafe_allow_html=True)

        # MEAL PLAN
        st.markdown('<div class="subsection-label">Sample daily meals</div>', unsafe_allow_html=True)
        meal_html = '<div class="meal-grid">'
        for meal_time, meal_content in pd["meals"].items():
            meal_html += f'<div class="meal-card"><div class="meal-card-header"><div class="meal-card-title">{meal_time}</div></div><div class="meal-card-body">{meal_content}</div></div>'
        meal_html += '</div>'
        st.markdown(meal_html, unsafe_allow_html=True)

        # TRACKING TIPS
        st.markdown("""
        <div class="tracking-grid" style="margin-top:14px;">
            <div class="tracking-title">Meal tracking essentials</div>
            <div class="tracking-items">
                <div class="tracking-item"><strong>Use MyFitnessPal or Cronometer</strong> — scan barcodes and log everything. Most people underestimate intake by 20-30% without tracking.</div>
                <div class="tracking-item"><strong>Weigh your food</strong> — especially calorie-dense foods like oils, nuts, rice, and peanut butter. Eyeballing is consistently inaccurate.</div>
                <div class="tracking-item"><strong>Meal prep on Sunday</strong> — cook proteins, grains, and vegetables in bulk. Removes daily decisions and keeps you consistent.</div>
                <div class="tracking-item"><strong>Hit protein first</strong> — plan every meal around your protein target. Carbs and fats fit around it. Protein is the most important macro.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # SUPPLEMENTS
        st.markdown("""
        <div class="supps-card">
            <div class="tracking-title">Evidence-based supplements</div>
            <div class="supps-grid">
                <div class="supp-item">
                    <span class="supp-name">Creatine Monohydrate</span>
                    <span class="supp-dose">5g per day · any time</span>
                    Most researched supplement. Improves strength and muscle gain. No loading phase needed.
                </div>
                <div class="supp-item">
                    <span class="supp-name">Protein Powder</span>
                    <span class="supp-dose">As needed for protein target</span>
                    Use whey (fastest absorbing) post-workout or to fill protein gaps in meals.
                </div>
                <div class="supp-item">
                    <span class="supp-name">Caffeine</span>
                    <span class="supp-dose">150-250mg · 30-60 min pre-training</span>
                    Improves performance and focus. Use sparingly — not every session.
                </div>
                <div class="supp-item">
                    <span class="supp-name">Multivitamin</span>
                    <span class="supp-dose">1 daily with a meal · optional</span>
                    Insurance against deficiencies. Not needed with a varied diet.
                </div>
            </div>
            <div class="macro-source" style="margin-top:14px;margin-bottom:0;">Test boosters, fat burners, and BCAAs are not strongly supported by current evidence.</div>
        </div>
        """, unsafe_allow_html=True)

        # TIPS
        st.markdown(f"""
        <div class="tips-section">
            <div class="tips-title">Key principles for {pd['discipline']}</div>
        """, unsafe_allow_html=True)
        tips_html = ""
        for i, tip in enumerate(pd["tips"]):
            tips_html += f'<div class="tip-item"><div class="tip-num">0{i+1}</div><div class="tip-text">{tip}</div></div>'
        st.markdown(tips_html + "</div></div></div>", unsafe_allow_html=True)

        # DOWNLOAD
        plan_text = f"FitByMinh — {pd['discipline']} Plan\n{pd['source']}\n\n"
        plan_text += f"PERSONALIZED MACROS:\nCalories: {nut['calories']} kcal | Protein: {nut['protein_g']}g | Carbs: {nut['carb_g']}g | Fat: {nut['fat_g']}g\n\nWEEKLY SCHEDULE:\n"
        for day, data in pd["week"].items():
            if data["type"] == "training":
                plan_text += f"\n{day} — {data['focus']}\n"
                for ex, sets, rest, yt in data["exercises"]:
                    plan_text += f"  {ex}: {sets}, rest {rest}\n"
            else:
                plan_text += f"\n{day} — Rest Day\n"

        st.markdown('<div style="padding:20px 48px 60px;">', unsafe_allow_html=True)
        st.download_button("Download my plan", data=plan_text, file_name="fitbyminh_plan.txt", mime="text/plain")
        st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# TAB 2 — CHAT
# ============================================================
with tab2:
    if st.session_state.plan_data is None:
        st.markdown("""
        <div class="chat-wrap">
            <div class="empty-state">
                <div class="empty-icon">01</div>
                <div class="empty-title">No plan yet</div>
                <div class="empty-body">Generate your plan first in the Build my plan tab, then come back here to talk to your AI trainer.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        discipline = st.session_state.discipline or "Fitness"
        st.markdown(f"""
        <div class="chat-wrap">
            <div class="chat-header">
                <div class="chat-title">
                    {discipline} Trainer
                    <span class="chat-badge">AI Coach</span>
                </div>
                <div class="chat-sub">Ask about technique, swap exercises, adjust rest periods, dial in nutrition, or anything else about your training. Tutorial videos appear automatically when relevant.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        discipline_questions = {
            "Bodybuilding": {
                "Feel chest more on bench?": "I do not feel my chest when bench pressing. How do I improve mind-muscle connection?",
                "Best back width exercises?": "What are the best exercises for building a wider back?",
                "Swap an exercise?": "Can you suggest a replacement for the hack squat? I do not have access to one.",
                "When to do cardio?": "When is the best time to do cardio when training for bodybuilding?"
            },
            "Body Recomposition": {
                "How to know if in deficit?": "How do I know if I am actually in a caloric deficit for recomposition?",
                "Is fasted cardio better?": "Is fasted cardio more effective for body recomposition?",
                "Scale not moving — normal?": "The scale is not moving but I look different. Is this normal during recomposition?",
                "Swap an exercise?": "Can you suggest a replacement for the Bulgarian split squat? My balance is not good enough yet."
            },
            "Lose Weight": {
                "Not losing weight — why?": "I am in a caloric deficit but not losing weight. What could be causing this?",
                "Best cardio for fat loss?": "What type of cardio is most effective for fat loss?",
                "Swap an exercise?": "Can you give me an easier alternative to burpees? I find them very difficult.",
                "How to avoid muscle loss?": "How do I avoid losing muscle while I am in a caloric deficit?"
            },
            "Improve Endurance": {
                "Run without getting tired?": "How do I run longer without getting tired so quickly?",
                "What is Zone 2 training?": "Can you explain what Zone 2 training is and why it matters?",
                "Swap an exercise?": "Can I replace running with cycling for this program?",
                "How to breathe running?": "What is the correct breathing pattern when running?"
            },
            "Maintain Fitness": {
                "How to stay motivated?": "How do I stay motivated when I am not training for a specific goal?",
                "Can I change exercises?": "Can I swap any of the exercises in my plan for ones I enjoy more?",
                "How much protein needed?": "Do I still need a high protein intake when just maintaining fitness?",
                "Add more intensity?": "I want to make my maintenance sessions more challenging. What should I add?"
            }
        }

        questions = discipline_questions.get(discipline, {
            "How do I start?": "How do I get started with my training plan?",
            "What should I eat?": "What should I eat to support my training?",
            "How to track progress?": "How do I track my progress effectively?",
            "Swap an exercise?": "Can you suggest an alternative exercise for one in my plan?"
        })

        st.markdown('<div style="padding: 0 48px 16px;">', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        for col, (label, question) in zip([c1, c2, c3, c4], questions.items()):
            with col:
                if st.button(label, key=f"q_{label}"):
                    st.session_state.chat_history.append({"role": "user", "content": question})
                    st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div style="padding:0 48px;">', unsafe_allow_html=True)

        if not st.session_state.chat_history:
            st.markdown("""
            <div class="empty-state" style="margin-top:8px;">
                <div class="empty-icon">→</div>
                <div class="empty-title">Start the conversation</div>
                <div class="empty-body">Use the quick questions above or type your own. Ask about technique, swapping exercises, adjusting rest periods, or anything about your plan.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    st.markdown(f'<div class="msg-you"><div class="msg-you-label">You</div><div class="msg-you-body">{msg["content"]}</div></div>', unsafe_allow_html=True)
                else:
                    idx = st.session_state.chat_history.index(msg)
                    prev_q = st.session_state.chat_history[idx-1]["content"] if idx > 0 else ""
                    yt_links = get_youtube_links(prev_q)
                    yt_html = ""
                    if yt_links:
                        yt_html = '<div style="padding:0 18px 14px;">'
                        for title, url in yt_links:
                            yt_html += f'<a class="yt-link" href="{url}" target="_blank">Watch · {title}</a>'
                        yt_html += '</div>'
                    st.markdown(f'<div class="msg-ai"><div class="msg-ai-label">{discipline} Trainer</div><div class="msg-ai-body">{md_to_html(msg["content"])}</div>{yt_html}</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        user_msg = st.chat_input("Ask about technique, adjusting your plan, nutrition, or anything else...")
        if user_msg:
            st.session_state.chat_history.append({"role": "user", "content": user_msg})

            system_prompt = f"""You are an expert {discipline} coach. Answer ONLY what the user asked — nothing else.

Rules:
- If the user asks to adjust reps, sets, or rest: rewrite the COMPLETE weekly schedule with EVERY exercise updated. Include every day, every exercise. Do not add tips, nutrition, progressive overload, or any extra sections.
- If the user asks to swap one exercise: give the replacement and one sentence why. Nothing else.
- If the user asks about technique: numbered cues only. Nothing else.
- If the user asks about nutrition: numbers only (calories, grams). Nothing else.
- Never add sections the user did not ask for.
- Always complete the full response — never stop mid-list or mid-sentence.

User profile: {st.session_state.user_profile}"""

            history_window = st.session_state.chat_history[-6:]
            api_messages = [{"role": m["role"], "content": m["content"]} for m in history_window]

            with st.spinner("Your trainer is thinking..."):
                local_model, local_tokenizer = load_local_model()
                if local_model is not None:
                    try:
                        reply = generate_response_local(local_model, local_tokenizer, system_prompt, api_messages)
                    except Exception as e:
                        import traceback
                        reply = f"Error: {type(e).__name__}: {e}\n\n```\n{traceback.format_exc()}\n```"
                else:
                    reply = "Local model not found. Please run train_model.py first."

            st.session_state.chat_history.append({"role": "assistant", "content": reply})
            st.rerun()

        if st.session_state.chat_history:
            st.markdown('<div style="padding:16px 48px 60px;">', unsafe_allow_html=True)
            if st.button("Clear conversation"):
                st.session_state.chat_history = []
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)