# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SAAM (Sustainability Aware Asset Management) course project — Portfolio Allocation with a Carbon Objective.

- **Group:** BR — **North America + Europe / Scope 1+2**
- **Part I deadline:** April 12, 2026 (preliminary results)
- **Part II deadline:** May 29, 2026 (complete package)

## Key Details

- Data covers 2545 firms, from end 1999 to end 2025
- Allocation exercise starts at end of 2013, using 10 years of monthly returns (2004–2013) for initial estimation
- Portfolio rebalanced annually (Dec 2013 to Dec 2024), performance computed monthly
- Long-only (non-negative weights) minimum-variance optimization
- CO2 emissions: Scope 1 + Scope 2 combined (our group's assignment)
- Region filter: North America + Europe

## Commands

- **Run:** `python main.py`
- **Virtual environment:** `.venv/` (activate with `.venv/Scripts/activate` on Windows)
- **Install dependencies:** `pip install -r requirements.txt`
