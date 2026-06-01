"""Export a HOPS fplot PDF for one fringe file."""

import argparse

from eht_inspection.fringe import export_fplot_pdf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("fringe_file")
    parser.add_argument("--outdir", default="pdf")
    args = parser.parse_args()
    pdf_path = export_fplot_pdf(args.fringe_file, outdir=args.outdir)
    print(pdf_path)

if __name__ == "__main__":
    main()
