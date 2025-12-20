"""Entry point for the AstroStack live stacking app."""

from astrostack.live_stack_app import main


if __name__ == "__main__":
    main(outdir="captures", roi=1024, binning=1)
