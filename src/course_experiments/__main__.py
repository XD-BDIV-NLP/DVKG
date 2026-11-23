from __future__ import annotations

import argparse
import json
from pathlib import Path

from .manifest import freeze, read_document, validate
from .matrix import generate


def main() -> None:
    parser = argparse.ArgumentParser(description='Offline experiment input freezing and matrix')
    commands = parser.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('freeze')
    for name in ('courses', 'experiments', 'sources', 'assets', 'output'):
        prepare.add_argument('--' + name, required=True)
    prepare.add_argument('--stage', choices=['construction', 'evaluation'], required=True)
    check = commands.add_parser('validate')
    check.add_argument('--manifest', required=True)
    matrix = commands.add_parser('matrix')
    matrix.add_argument('--manifest', required=True)
    matrix.add_argument('--output', required=True)
    args = parser.parse_args()
    try:
        if args.command == 'freeze':
            frozen = freeze(read_document(args.courses), read_document(args.experiments),
                            read_document(args.sources), read_document(args.assets), output=args.output,
                            config_paths=[args.courses, args.experiments, args.sources, args.assets], stage=args.stage)
            print(frozen['manifest_id'])
        else:
            frozen = validate(args.manifest)
            if args.command == 'validate':
                print(frozen['manifest_id'])
            else:
                with Path(args.output).open('x', encoding='utf-8') as handle:
                    json.dump(generate(frozen), handle, ensure_ascii=False, indent=2)
                    handle.write('\n')
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f'{exc}\n')


if __name__ == '__main__':
    main()
