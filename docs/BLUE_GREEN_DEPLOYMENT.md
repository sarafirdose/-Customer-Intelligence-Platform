# Blue/Green Deployment Guide

## Overview
Blue/Green deployment maintains two identical production environments (BLUE and GREEN) for zero-downtime releases and instant traffic switching.

## Architecture
- **BLUE**: Active serving environment.
- **GREEN**: Staging/Candidate environment.

## Switching Environments
Trigger instant traffic switch via API:
`POST /api/v1/deployment/bluegreen`

Or via Streamlit Operations/Deployment dashboard (Page 10).
