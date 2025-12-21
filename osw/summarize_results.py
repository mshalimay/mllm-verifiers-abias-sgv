import argparse
import json
from pathlib import Path

DOMAINS = [
    'chrome',
    'gimp',
    'libreoffice_calc',
    'libreoffice_impress',
    'libreoffice_writer',
    'multi_apps',
    'os',
    'thunderbird',
    'vlc',
    'vs_code'
]


def summarize_results(run_results_path: Path):
    results: dict[str, tuple[int, int]] = {}
    total_task_count = 0
    total_score = 0

    for domain in DOMAINS:
        domain_results_path = run_results_path / domain
        if not domain_results_path.exists():
            print(f'`{domain}` results are not available yet.')
            continue
        task_count = 0
        total_domain_score = 0
        for trajectory_file in domain_results_path.rglob('trajectory.json'):
            trajectory = json.loads(
                trajectory_file.read_text(encoding='utf-8'))
            score = trajectory.get('score')
            if score is not None:
                task_count += 1
                total_domain_score += score
        if task_count == 0:
            print(f'`{domain}` results are not available yet.')
            continue
        results[domain] = (task_count, total_domain_score)
        total_task_count += task_count
        total_score += total_domain_score
    lines = [f'Total: {total_score:.2f}/{total_task_count} '
             f'({total_score / total_task_count:.1%})', '']
    for domain, (task_count, total_domain_score) in results.items():
        lines.append(f'{domain}: {total_domain_score:.2f}/{task_count} '
                     f'({total_domain_score / task_count:.1%})')
    print('--------------------------')
    print('\n'.join(lines))
    print('--------------------------')
    summary_path = run_results_path / 'summary.txt'
    summary_path.write_text('\n'.join(lines))
    print(f'Summary saved to {summary_path}.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_name')
    args = parser.parse_args()
    results_path = (Path.cwd() / 'results' / 'pyautogui' / 'screenshot')
    run_results_path = results_path / args.run_name
    if not run_results_path.is_dir():
        run_results_path = next(
            subdirectory_path for subdirectory_path in results_path.iterdir()
            if subdirectory_path.name.startswith(args.run_name)
        )
    summarize_results(run_results_path)


if __name__ == '__main__':
    main()
