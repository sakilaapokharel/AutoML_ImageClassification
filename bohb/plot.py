import matplotlib.pyplot as plt


def extract_bohb_curve(results):
    """
    Extract BOHB incumbent curve.

    X-axis:
        wall clock time since first run

    Y-axis:
        best validation accuracy seen so far
    """

    points = []

    runs = results.get_all_runs()

    for run in runs:

        if run.info is None:
            continue

        if "val_accuracy" not in run.info:
            continue

        if "finished" not in run.time_stamps:
            continue


        finish_time = run.time_stamps["finished"]

        accuracy = run.info["val_accuracy"]


        points.append(
            (
                finish_time,
                accuracy
            )
        )


    if len(points) == 0:
        return [], []


    # sort by wall clock time
    points.sort(
        key=lambda x: x[0]
    )


    start_time = points[0][0]


    times = []
    best_acc = []

    current_best = 0


    for t, acc in points:

        elapsed = t - start_time

        current_best = max(
            current_best,
            acc
        )

        times.append(
            elapsed / 60.0   # minutes
        )

        best_acc.append(
            current_best
        )


    return times, best_acc



def extract_portfolio_curve(portfolio_results):

    all_points = []


    for portfolio in portfolio_results:

        results = portfolio["results"]

        runs = results.get_all_runs()


        for run in runs:

            if run.info is None:
                continue


            if "val_accuracy" not in run.info:
                continue


            if "finished" not in run.time_stamps:
                continue


            all_points.append(
                (
                    run.time_stamps["finished"],
                    run.info["val_accuracy"]
                )
            )



    if len(all_points) == 0:
        return [], []


    all_points.sort(
        key=lambda x:x[0]
    )


    start_time = all_points[0][0]


    times=[]
    best_acc=[]

    current_best=0


    for t,acc in all_points:

        elapsed = t-start_time


        current_best=max(
            current_best,
            acc
        )


        times.append(
            elapsed/60.0
        )

        best_acc.append(
            current_best
        )


    return times,best_acc



def plot_comparison(
        full_results,
        portfolio_results,
        save_path
):


    full_time, full_acc = extract_bohb_curve(
        full_results
    )


    portfolio_time, portfolio_acc = extract_portfolio_curve(
        portfolio_results
    )



    plt.figure(
        figsize=(8,6)
    )


    plt.plot(
        full_time,
        full_acc,
        marker="o",
        label="Full BOHB"
    )


    plt.plot(
        portfolio_time,
        portfolio_acc,
        marker="o",
        label="Portfolio BOHB"
    )


    plt.xlabel(
        "Wall clock time (minutes)"
    )


    plt.ylabel(
        "Best validation accuracy"
    )


    plt.title(
        "BOHB Optimization Progress"
    )


    plt.grid()

    plt.legend()


    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()