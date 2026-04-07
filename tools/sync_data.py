import argparse
import shutil
from pathlib import Path


def copy_selected_csv_files(source_dir: Path, target_dirs: list[Path]) -> None:
	"""Copy key CSVs from OB_Radix to each target data directory."""
	csv_names = [
		"character_explanations.csv",
		"character_explanations_CN.csv",
		"character_analysis.csv",
		"radical_explanation.csv",
	]
	missing_files: list[str] = []
	for csv_name in csv_names:
		source_file = source_dir / csv_name
		if not source_file.exists():
			missing_files.append(csv_name)
			continue
		for target in target_dirs:
			target.mkdir(parents=True, exist_ok=True)
			shutil.copy2(source_file, target / csv_name)
	if missing_files:
		print("[warn] 未在源目录中找到以下文件:", ", ".join(missing_files))


def copy_optional_assets(source_dir: Path, target_dirs: list[Path]) -> None:
	"""Optionally mirror image/radical folders if needed for experiments."""
	folders = ["organized_radicals", "img_zi"]
	for folder in folders:
		src_path = source_dir / folder
		if not src_path.exists():
			continue
		for target in target_dirs:
			dst_path = target.parent / folder  # keep sibling to data/ if needed
			dst_path.mkdir(parents=True, exist_ok=True)
			shutil.copytree(src_path, dst_path, dirs_exist_ok=True)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Sync OB_Radix data to experiment data directories.")
	parser.add_argument("--src", type=str, default=str(Path("OB_Radix")), help="OB_Radix 源目录")
	parser.add_argument(
		"--targets",
		nargs="*",
		default=[str(Path("experiments") / "data")],
		help="目标 data 目录列表",
	)
	parser.add_argument("--with-assets", action="store_true", help="同时同步图片/部件目录（如 organized_radicals, img_zi）")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	source_dir = Path(args.src).resolve()
	target_dirs = [Path(p).resolve() for p in args.targets]
	print(f"[info] 源目录: {source_dir}")
	print("[info] 目标目录:")
	for d in target_dirs:
		print(f"  - {d}")
	if not source_dir.exists():
		raise FileNotFoundError(f"源目录不存在: {source_dir}")

	copy_selected_csv_files(source_dir, target_dirs)
	if args.with_assets:
		copy_optional_assets(source_dir, target_dirs)
	print("[done] 数据同步完成。")


if __name__ == "__main__":
	main()
