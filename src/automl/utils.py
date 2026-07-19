def print_header(text):
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)


def print_step(text):
    print(f"\n[{text}]")


def print_progress(current, total, every=100):
    if current % every == 0 or current == total:
        print(f"## {current}/{total}")
