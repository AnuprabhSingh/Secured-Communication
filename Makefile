.PHONY: help setup run test clean view-results

help:
	@echo "THz Anti-Jamming Project - Available Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make setup          Create virtual environment and install dependencies"
	@echo ""
	@echo "Run Experiments:"
	@echo "  make run            Full paper reproduction (30-60 min, 5 seeds)"
	@echo "  make test           Quick test (2-3 min, fewer seeds)"
	@echo ""
	@echo "Results:"
	@echo "  make view-results   View results summary from JSON"
	@echo "  make clean          Remove outputs (keep code)"
	@echo ""
	@echo "Documentation:"
	@echo "  make readme         Open README.md"
	@echo "  make setup-guide    Open SETUP.md"
	@echo "  make results-guide  Open RESULTS.md"

setup:
	@echo "Creating virtual environment..."
	python3 -m venv .venv
	@echo "Activating and installing dependencies..."
	. .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt
	@echo "✓ Setup complete. Run: source .venv/bin/activate && make run"

run:
	@echo "Running full paper reproduction..."
	@. .venv/bin/activate && python run_all.py

test:
	@echo "Running quick test (2-3 min)..."
	@. .venv/bin/activate && python scripts/run_paper_fast.py

view-results:
	@echo "Results from outputs_joint_ao/:"
	@python3 << 'EOF'
import json
try:
    with open("outputs_joint_ao/paper_results.json") as f:
        data = json.load(f)
        print("\n" + "="*70)
        print("FUZZY WOLF-PHC RESULTS".center(70))
        print("="*70 + "\n")
        summary = data["summary"]
        for method in ["ao_baseline", "q_learning", "dqn", "fuzzy_wolf_phc"]:
            if method in summary:
                stats = summary[method]
                name = method.replace("_", " ").title()
                print(f"{name}:")
                print(f"  Rate:       {stats['rate_mean']:6.2f} ± {stats['rate_std']:.2f} bits/s/Hz")
                print(f"  Protection: {stats['protection_mean']:5.1f}% ± {stats['protection_std']:.1f}%")
                if method == "fuzzy_wolf_phc":
                    print("  ⭐ PROPOSED METHOD\n")
                else:
                    print()
        print("="*70)
except FileNotFoundError:
    print("❌ Results not found. Run: make run")
EOF

clean:
	@echo "Removing results..."
	rm -rf outputs_joint_ao/paper_results.json outputs_joint_ao/ieee_plots
	@echo "✓ Cleaned. Code preserved."

readme:
	@command -v open >/dev/null && open README.md || cat README.md | head -50

setup-guide:
	@command -v open >/dev/null && open SETUP.md || cat SETUP.md | head -50

results-guide:
	@command -v open >/dev/null && open RESULTS.md || cat RESULTS.md | head -50
