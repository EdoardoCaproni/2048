"""Copy the figures used by the report from results/ to report/figures/.
Run from the project root:  python experiments/collect_figures.py
"""
import shutil

from common import RESULTS, ROOT

DEST = ROOT / 'report' / 'figures'


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    n = 0
    for path in sorted(RESULTS.rglob('fig_*.pdf')):
        shutil.copy(path, DEST / path.name)
        n += 1
    print(f'{n} figures copied to {DEST}')


if __name__ == '__main__':
    main()
