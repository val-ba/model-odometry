import optuna

from optuna.trial import TrialState
import argparse


parser = argparse.ArgumentParser(description='Load Optuna study')
parser.add_argument('--study_name', "-n", type=str, default="odometry_study1", help='Name of the study')



if __name__ == "__main__":
    study_name = parser.parse_args().study_name
    study = optuna.create_study(direction="minimize", study_name=study_name, storage="sqlite:///"+study_name+".db", load_if_exists=True)
    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    optuna.visualization.plot_optimization_history(study).show()
    optuna.visualization.plot_slice(study).show()
    optuna.visualization.plot_parallel_coordinate(study).show()
    optuna.visualization.plot_contour(study, params=["number_of_layers", "architecture"]).show()
    optuna.visualization.plot_param_importances(study).show()
    optuna.visualization.plot_edf(study).show()
    optuna.visualization.plot_intermediate_values(study).show()
    optuna.visualization.plot_optimization_history(study).show()


    
    print("Plots saved to plots folder")
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))
    print("Best trial:")
    trial = study.best_trial
