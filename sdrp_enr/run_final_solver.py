"""Run the final four-generator, two-neighborhood configuration."""
from .run_ablation import main as run_experiments

def main():
    run_experiments(full_only=True)

if __name__ == "__main__":
    main()
