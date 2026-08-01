def print_header(text):
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)


def print_step(text):
    print(f"\n[{text}]")


def print_progress(current, total, every=100):
    if current % every == 0 or current == total:
        print(f"## {current}/{total}")


def search_space_size(search_space):
    size = 1

    for values in search_space.values():
        size *= len(values)

    return size


def estimate_bohb_trials(search_space):

    n_configs = search_space_size(search_space)

    if n_configs <= 100:
        return max(30, n_configs // 2)

    elif n_configs <= 1000:
        return 100

    elif n_configs <= 10000:
        return 200

    else:
        return 300
