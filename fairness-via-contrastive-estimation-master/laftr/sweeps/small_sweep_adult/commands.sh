python src/run_laftr.py sweeps/small_sweep_adult/config.json --dirs local --data adult -o model.class=WeightedDemParWassGan,model.fair_coeff=0.1,exp_name="small_sweep_adult/data--adult--model_class-WeightedDemParWassGan--model_fair_coeff-0_1"
python src/run_laftr.py sweeps/small_sweep_adult/config.json --dirs local --data adult -o model.class=WeightedDemParWassGan,model.fair_coeff=0.2,exp_name="small_sweep_adult/data--adult--model_class-WeightedDemParWassGan--model_fair_coeff-0_2"
python src/run_laftr.py sweeps/small_sweep_adult/config.json --dirs local --data adult -o model.class=WeightedDemParWassGan,model.fair_coeff=0.3,exp_name="small_sweep_adult/data--adult--model_class-WeightedDemParWassGan--model_fair_coeff-0_3"
python src/run_laftr.py sweeps/small_sweep_adult/config.json --dirs local --data adult -o model.class=WeightedEqoddsWassGan,model.fair_coeff=0.1,exp_name="small_sweep_adult/data--adult--model_class-WeightedEqoddsWassGan--model_fair_coeff-0_1"
python src/run_laftr.py sweeps/small_sweep_adult/config.json --dirs local --data adult -o model.class=WeightedEqoddsWassGan,model.fair_coeff=0.2,exp_name="small_sweep_adult/data--adult--model_class-WeightedEqoddsWassGan--model_fair_coeff-0_2"
python src/run_laftr.py sweeps/small_sweep_adult/config.json --dirs local --data adult -o model.class=WeightedEqoddsWassGan,model.fair_coeff=0.3,exp_name="small_sweep_adult/data--adult--model_class-WeightedEqoddsWassGan--model_fair_coeff-0_3"
python src/run_laftr.py sweeps/small_sweep_adult/config.json --dirs local --data adult -o model.class=WeightedEqoppWassGan,model.fair_coeff=0.1,exp_name="small_sweep_adult/data--adult--model_class-WeightedEqoppWassGan--model_fair_coeff-0_1"
python src/run_laftr.py sweeps/small_sweep_adult/config.json --dirs local --data adult -o model.class=WeightedEqoppWassGan,model.fair_coeff=0.2,exp_name="small_sweep_adult/data--adult--model_class-WeightedEqoppWassGan--model_fair_coeff-0_2"
python src/run_laftr.py sweeps/small_sweep_adult/config.json --dirs local --data adult -o model.class=WeightedEqoppWassGan,model.fair_coeff=0.3,exp_name="small_sweep_adult/data--adult--model_class-WeightedEqoppWassGan--model_fair_coeff-0_3"
python src/run_laftr.py sweeps/small_sweep_adult/config.json --dirs local --data adult -o model.class=DemParGan,model.fair_coeff=0.1,exp_name="small_sweep_adult/data--adult--model_class-DemParGan--model_fair_coeff-0_1"
python src/run_laftr.py sweeps/small_sweep_adult/config.json --dirs local --data adult -o model.class=DemParGan,model.fair_coeff=0.2,exp_name="small_sweep_adult/data--adult--model_class-DemParGan--model_fair_coeff-0_2"
python src/run_laftr.py sweeps/small_sweep_adult/config.json --dirs local --data adult -o model.class=DemParGan,model.fair_coeff=0.3,exp_name="small_sweep_adult/data--adult--model_class-DemParGan--model_fair_coeff-0_3"
python src/run_unf_clf.py sweeps/small_sweep_adult/config.json --dirs local --data adult -o model.class=WeightedDemParWassGan,model.fair_coeff=0.1,transfer.epoch_number=0,transfer.model_seed=1,transfer.repr_phase=Test,exp_name="small_sweep_adult/data--adult--model_class-WeightedDemParWassGan--model_fair_coeff-0_1/transfer_epoch_number-0--transfer_model_seed-1--transfer_repr_phase-Test"
python src/run_unf_clf.py sweeps/small_sweep_adult/config.json --dirs local --data adult -o model.class=WeightedDemParWassGan,model.fair_coeff=0.1,transfer.epoch_number=0,transfer.model_seed=1,transfer.repr_phase=Valid,exp_name="small_sweep_adult/data--adult--model_class-WeightedDemParWassGan--model_fair_coeff-0_1/transfer_epoch_number-0--transfer_model_seed-1--transfer_repr_phase-Valid"
python src/run_unf_clf.py sweeps/small_sweep_adult/config.json --dirs local --data adult -o model.class=WeightedDemParWassGan,model.fair_coeff=0.1,transfer.epoch_number=0,transfer.model_seed=2,transfer.repr_phase=Test,exp_name="small_sweep_adult/data--adult--model_class-WeightedDemParWassGan--model_fair_coeff-0_1/transfer_epoch_number-0--transfer_model_seed-2--transfer_repr_phase-Test"
python src/run_unf_clf.py sweeps/small_sweep_adult/config.json --dirs local --data adult -o model.class=WeightedDemParWassGan,model.fair_coeff=0.1,transfer.epoch_number=0,transfer.model_seed=2,transfer.repr_phase=Valid,exp_name="small_sweep_adult/data--adult--model_class-WeightedDemParWassGan--model_fair_coeff-0_1/transfer_epoch_number-0--transfer_model_seed-2--transfer_repr_phase-Valid"
python src/run_unf_clf.py sweeps/small_sweep_adult/config.json --dirs local --data adult -o model.class=WeightedDemParWassGan,model.fair_coeff=0.1,transfer.epoch_number=0,transfer.model_seed=3,transfer.repr_phase=Test,exp_name="small_sweep_adult/data--adult--model_class-WeightedDemParWassGan--model_fair_coeff-0_1/transfer_epoch_number-0--transfer_model_seed-3--transfer_repr_phase-Test"
python src/run_unf_clf.py sweeps/small_sweep_adult/config.json --dirs local --data adult -o model.class=WeightedDemParWassGan,model.fair_coeff=0.1,transfer.epoch_number=0,transfer.model_seed=3,transfer.repr_phase=Valid,exp_name="small_sweep_adult/data--adult--model_class-WeightedDemParWassGan--model_fair_coeff-0_1/transfer_epoch_number-0--transfer_model_seed-3--transfer_repr_phase-Valid"
python src/run_unf_clf.py sweeps/small_sweep_adult/config.json --dirs local --data adult -o model.class=WeightedDemParWassGan,model.fair_coeff=0.2,transfer.epoch_number=0,transfer.model_seed=1,transfer.repr_phase=Test,exp_name="small_sweep_adult/data--adult--model_class-WeightedDemParWassGan--model_fair_coeff-0_2/transfer_epoch_number-0--transfer_model_seed-1--transfer_repr_phase-Test"
python src/run_unf_clf.py sweeps/small_sweep_adult/config.json --dirs local --data adult -o model.class=WeightedDemParWassGan,model.fair_coeff=0.2,transfer.epoch_number=0,transfer.model_seed=1,transfer.repr_phase=Valid,exp_name="small_sweep_adult/data--adult--model_class-WeightedDemParWassGan--model_fair_coeff-0_2/transfer_epoch_number-0--transfer_model_seed-1--transfer_repr_phase-Valid"
python src/run_unf_clf.py sweeps/small_sweep_adult/config.json --dirs local --data adult -o model.class=WeightedDemParWassGan,model.fair_coeff=0.2,transfer.epoch_number=0,transfer.model_seed=2,transfer.repr_phase=Test,exp_name="small_sweep_adult/data--adult--model_class-WeightedDemParWassGan--model_fair_coeff-0_2/transfer_epoch_number-0--transfer_model_seed-2--transfer_repr_phase-Test"
python src/run_unf_clf.py sweeps/small_sweep_adult/config.json --dirs local --data adult -o model.class=WeightedDemParWassGan,model.fair_coeff=0.2,transfer.epoch_number=0,transfer.model_seed=2,transfer.repr_phase=Valid,exp_name="small_sweep_adult/data--adult--model_class-WeightedDemParWassGan--model_fair_coeff-0_2/transfer_epoch_number-0--transfer_model_seed-2--transfer_repr_phase-Valid"
python src/run_unf_clf.py sweeps/small_sweep_adult/config.json --dirs local --data adult -o model.class=WeightedDemParWassGan,model.fair_coeff=0.2,transfer.epoch_number=0,transfer.model_seed=3,transfer.repr_phase=Test,exp_name="small_sweep_adult/data--adult--model_class-WeightedDemParWassGan--model_fair_coeff-0_2/transfer_epoch_number-0--transfer_model_seed-3--transfer_repr_phase-Test"
python src/run_unf_clf.py sweeps/small_sweep_adult/config.json --dirs local --data adult -o model.class=WeightedDemParWassGan,model.fair_coeff=0.2,transfer.epoch_number=0,transfer.model_seed=3,transfer.repr_phase=Valid,exp_name="small_sweep_adult/data--adult--model_class-WeightedDemParWassGan--model_fair_coeff-0_2/transfer_epoch_number-0--transfer_model_seed-3--transfer_repr_phase-Valid"
python src/run_unf_clf.py sweeps/small_sweep_adult/config.json --dirs local --data adult -o model.class=WeightedDemParWassGan,model.fair_coeff=0.3,transfer.epoch_number=0,transfer.model_seed=1,transfer.repr_phase=Test,exp_name="small_sweep_adult/data--adult--model_class-WeightedDemParWassGan--model_fair_coeff-0_3/transfer_epoch_number-0--transfer_model_seed-1--transfer_repr_phase-Test"
python src/run_unf_clf.py sweeps/small_sweep_adult/config.json --dirs local --data adult -o model.class=WeightedDemParWassGan,model.fair_coeff=0.3,transfer.epoch_number=0,transfer.model_seed=1,transfer.repr_phase=Valid,exp_name="small_sweep_adult/data--adult--model_class-WeightedDemParWassGan--model_fair_coeff-0_3/transfer_epoch_number-0--transfer_model_seed-1--transfer_repr_phase-Valid"
python src/run_unf_clf.py sweeps/small_sweep_adult/config.json --dirs local --data adult -o model.class=WeightedDemParWassGan,model.fair_coeff=0.3,transfer.epoch_number=0,transfer.model_seed=2,transfer.repr_phase=Test,exp_name="small_sweep_adult/data--adult--model_class-WeightedDemParWassGan--model_fair_coeff-0_3/transfer_epoch_number-0--transfer_model_seed-2--transfer_repr_phase-Test"
python src/run_unf_clf.py sweeps/small_sweep_adult/config.json --dirs local --data adult -o model.class=WeightedDemParWassGan,model.fair_coeff=0.3,transfer.epoch_number=0,transfer.model_seed=2,transfer.repr_phase=Valid,exp_name="small_sweep_adult/data--adult--model_class-WeightedDemParWassGan--model_fair_coeff-0_3/transfer_epoch_number-0--transfer_model_seed-2--transfer_repr_phase-Valid"
python src/run_unf_clf.py sweeps/small_sweep_adult/config.json --dirs local --data adult -o model.class=WeightedDemParWassGan,model.fair_coeff=0.3,transfer.epoch_number=0,transfer.model_seed=3,transfer.repr_phase=Test,exp_name="small_sweep_adult/data--adult--model_class-WeightedDemParWassGan--model_fair_coeff-0_3/transfer_epoch_number-0--transfer_model_seed-3--transfer_repr_phase-Test"
python src/run_unf_clf.py sweeps/small_sweep_adult/config.json --dirs local --data adult -o model.class=WeightedDemParWassGan,model.fair_coeff=0.3,transfer.epoch_number=0,transfer.model_seed=3,transfer.repr_phase=Valid,exp_name="small_sweep_adult/data--adult--model_class-WeightedDemParWassGan--model_fair_coeff-0_3/transfer_epoch_number-0--transfer_model_seed-3--transfer_repr_phase-Valid"
python src/run_unf_clf.py sweeps/small_sweep_adult/config.json --dirs local --data adult -o model.class=WeightedEqoddsWassGan,model.fair_coeff=0.1,transfer.epoch_number=0,transfer.model_seed=1,transfer.repr_phase=Test,exp_name="small_sweep_adult/data--adult--model_class-WeightedEqoddsWassGan--model_fair_coeff-0_1/transfer_epoch_number-0--transfer_model_seed-1--transfer_repr_phase-Test"
python src/run_unf_clf.py sweeps/small_sweep_adult/config.json --dirs local --data adult -o model.class=WeightedEqoddsWassGan,model.fair_coeff=0.1,transfer.epoch_number=0,transfer.model_seed=1,transfer.repr_phase=Valid,exp_name="small_sweep_adult/data--adult--model_class-WeightedEqoddsWassGan--model_fair_coeff-0_1/transfer_epoch_number-0--transfer_model_seed-1--transfer_repr_phase-Valid"
python src/run_unf_clf.py sweeps/small_sweep_adult/config.json --dirs local --data adult -o model.class=WeightedEqoddsWassGan,model.fair_coeff=0.1,transfer.epoch_number=0,transfer.model_seed=2,transfer.repr_phase=Test,exp_name="small_sweep_adult/data--adult--model_class-WeightedEqoddsWassGan--model_fair_coeff-0_1/transfer_epoch_number-0--transfer_model_seed-2--transfer_repr_phase-Test"
python src/run_unf_clf.py sweeps/small_sweep_adult/config.json --dirs local --data adult -o model.class=WeightedEqoddsWassGan,model.fair_coeff=0.1,transfer.epoch_number=0,transfer.model_seed=2,transfer.repr_phase=Valid,exp_name="small_sweep_adult/data--adult--model_class-WeightedEqoddsWassGan--model_fair_coeff-0_1/transfer_epoch_number-0--transfer_model_seed-2--transfer_repr_phase-Valid"
python src/run_unf_clf.py sweeps/small_sweep_adult/config.json --dirs local --data adult -o model.class=WeightedEqoddsWassGan,model.fair_coeff=0.1,transfer.epoch_number=0,transfer.model_seed=3,transfer.repr_phase=Test,exp_name="small_sweep_adult/data--adult--model_class-WeightedEqoddsWassGan--model_fair_coeff-0_1/transfer_epoch_number-0--transfer_model_seed-3--transfer_repr_phase-Test"
python src/run_unf_clf.py sweeps/small_sweep_adult/config.json --dirs local --data adult -o model.class=WeightedEqoddsWassGan,model.fair_coeff=0.1,transfer.epoch_number=0,transfer.model_seed=3,transfer.repr_phase=Valid,exp_name="small_sweep_adult/data--adult--model_class-WeightedEqoddsWassGan--model_fair_coeff-0_1/transfer_epoch_number-0--transfer_model_seed-3--transfer_repr_phase-Valid"
python src/run_unf_clf.py sweeps/small_sweep_adult/config.json --dirs local --data adult -o model.class=WeightedEqoddsWassGan,model.fair_coeff=0.2,transfer.epoch_number=0,transfer.model_seed=1,transfer.repr_phase=Test,exp_name="small_sweep_adult/data--adult--model_class-WeightedEqoddsWassGan--model_fair_coeff-0_2/transfer_epoch_number-0--transfer_model_seed-1--transfer_repr_phase-Test"
python src/run_unf_clf.py sweeps/small_sweep_adult/config.json --dirs local --data adult -o model.class=WeightedEqoddsWassGan,model.fair_coeff=0.2,transfer.epoch_number=0,transfer.model_seed=1,transfer.repr_phase=Valid,exp_name="small_sweep_adult/data--adult--model_class-WeightedEqoddsWassGan--model_fair_coeff-0_2/transfer_epoch_number-0--transfer_model_seed-1--transfer_repr_phase-Valid"
python src/run_unf_clf.py sweeps/small_sweep_adult/config.json --dirs local --data adult -o model.class=WeightedEqoddsWassGan,model.fair_coeff=0.2,transfer.epoch_number=0,transfer.model_seed=2,transfer.repr_phase=Test,exp_name="small_sweep_adult/data--adult--model_class-WeightedEqoddsWassGan--model_fair_coeff-0_2/transfer_epoch_number-0--transfer_model_seed-2--transfer_repr_phase-Test"
python src/run_unf_clf.py sweeps/small_sweep_adult/config.json --dirs local --data adult -o model.class=WeightedEqoddsWassGan,model.fair_coeff=0.2,transfer.epoch_number=0,transfer.model_seed=2,transfer.repr_phase=Valid,exp_name="small_sweep_adult/data--adult--model_class-WeightedEqoddsWassGan--model_fair_coeff-0_2/transfer_epoch_number-0--transfer_model_seed-2--transfer_repr_phase-Valid"
python src/run_unf_clf.py sweeps/small_sweep_adult/config.json --dirs local --data adult -o model.class=WeightedEqoddsWassGan,model.fair_coeff=0.2,transfer.epoch_number=0,transfer.model_seed=3,transfer.repr_phase=Test,exp_name="small_sweep_adult/data--adult--model_class-WeightedEqoddsWassGan--model_fair_coeff-0_2/transfer_epoch_number-0--transfer_model_seed-3--transfer_repr_phase-Test"
python src/run_unf_clf.py sweeps/small_sweep_adult/config.json --dirs local --data adult -o model.class=WeightedEqoddsWassGan,model.fair_coeff=0.2,transfer.epoch_number=0,transfer.model_seed=3,transfer.repr_phase=Valid,exp_name="small_sweep_adult/data--adult--model_class-WeightedEqoddsWassGan--model_fair_coeff-0_2/transfer_epoch_number-0--transfer_model_seed-3--transfer_repr_phase-Valid"
python src/run_unf_clf.py sweeps/small_sweep_adult/config.json --dirs local --data adult -o model.class=WeightedEqoddsWassGan,model.fair_coeff=0.3,transfer.epoch_number=0,transfer.model_seed=1,transfer.repr_phase=Test,exp_name="small_sweep_adult/data--adult--model_class-WeightedEqoddsWassGan--model_fair_coeff-0_3/transfer_epoch_number-0--transfer_model_seed-1--transfer_repr_phase-Test"
python src/run_unf_clf.py sweeps/small_sweep_adult/config.json --dirs local --data adult -o model.class=WeightedEqoddsWassGan,model.fair_coeff=0.3,transfer.epoch_number=0,transfer.model_seed=1,transfer.repr_phase=Valid,exp_name="small_sweep_adult/data--adult--model_class-WeightedEqoddsWassGan--model_fair_coeff-0_3/transfer_epoch_number-0--transfer_model_seed-1--transfer_repr_phase-Valid"
python src/run_unf_clf.py sweeps/small_sweep_adult/config.json --dirs local --data adult -o model.class=WeightedEqoddsWassGan,model.fair_coeff=0.3,transfer.epoch_number=0,transfer.model_seed=2,transfer.repr_phase=Test,exp_name="small_sweep_adult/data--adult--model_class-WeightedEqoddsWassGan--model_fair_coeff-0_3/transfer_epoch_number-0--transfer_model_seed-2--transfer_repr_phase-Test"
python src/run_unf_clf.py sweeps/small_sweep_adult/config.json --dirs local --data adult -o model.class=WeightedEqoddsWassGan,model.fair_coeff=0.3,transfer.epoch_number=0,transfer.model_seed=2,transfer.repr_phase=Valid,exp_name="small_sweep_adult/data--adult--model_class-WeightedEqoddsWassGan--model_fair_coeff-0_3/transfer_epoch_number-0--transfer_model_seed-2--transfer_repr_phase-Valid"
python src/run_unf_clf.py sweeps/small_sweep_adult/config.json --dirs local --data adult -o model.class=WeightedEqoddsWassGan,model.fair_coeff=0.3,transfer.epoch_number=0,transfer.model_seed=3,transfer.repr_phase=Test,exp_name="small_sweep_adult/data--adult--model_class-WeightedEqoddsWassGan--model_fair_coeff-0_3/transfer_epoch_number-0--transfer_model_seed-3--transfer_repr_phase-Test"
python src/run_unf_clf.py sweeps/small_sweep_adult/config.json --dirs local --data adult -o model.class=WeightedEqoddsWassGan,model.fair_coeff=0.3,transfer.epoch_number=0,transfer.model_seed=3,transfer.repr_phase=Valid,exp_name="small_sweep_adult/data--adult--model_class-WeightedEqoddsWassGan--model_fair_coeff-0_3/transfer_epoch_number-0--transfer_model_seed-3--transfer_repr_phase-Valid"
python src/run_unf_clf.py sweeps/small_sweep_adult/config.json --dirs local --data adult -o model.class=WeightedEqoppWassGan,model.fair_coeff=0.1,transfer.epoch_number=0,transfer.model_seed=1,transfer.repr_phase=Test,exp_name="small_sweep_adult/data--adult--model_class-WeightedEqoppWassGan--model_fair_coeff-0_1/transfer_epoch_number-0--transfer_model_seed-1--transfer_repr_phase-Test"
python src/run_unf_clf.py sweeps/small_sweep_adult/config.json --dirs local --data adult -o model.class=WeightedEqoppWassGan,model.fair_coeff=0.1,transfer.epoch_number=0,transfer.model_seed=1,transfer.repr_phase=Valid,exp_name="small_sweep_adult/data--adult--model_class-WeightedEqoppWassGan--model_fair_coeff-0_1/transfer_epoch_number-0--transfer_model_seed-1--transfer_repr_phase-Valid"
python src/run_unf_clf.py sweeps/small_sweep_adult/config.json --dirs local --data adult -o model.class=WeightedEqoppWassGan,model.fair_coeff=0.1,transfer.epoch_number=0,transfer.model_seed=2,transfer.repr_phase=Test,exp_name="small_sweep_adult/data--adult--model_class-WeightedEqoppWassGan--model_fair_coeff-0_1/transfer_epoch_number-0--transfer_model_seed-2--transfer_repr_phase-Test"
python src/run_unf_clf.py sweeps/small_sweep_adult/config.json --dirs local --data adult -o model.class=WeightedEqoppWassGan,model.fair_coeff=0.1,transfer.epoch_number=0,transfer.model_seed=2,transfer.repr_phase=Valid,exp_name="small_sweep_adult/data--adult--model_class-WeightedEqoppWassGan--model_fair_coeff-0_1/transfer_epoch_number-0--transfer_model_seed-2--transfer_repr_phase-Valid"
python src/run_unf_clf.py sweeps/small_sweep_adult/config.json --dirs local --data adult -o model.class=WeightedEqoppWassGan,model.fair_coeff=0.1,transfer.epoch_number=0,transfer.model_seed=3,transfer.repr_phase=Test,exp_name="small_sweep_adult/data--adult--model_class-WeightedEqoppWassGan--model_fair_coeff-0_1/transfer_epoch_number-0--transfer_model_seed-3--transfer_repr_phase-Test"
python src/run_unf_clf.py sweeps/small_sweep_adult/config.json --dirs local --data adult -o model.class=WeightedEqoppWassGan,model.fair_coeff=0.1,transfer.epoch_number=0,transfer.model_seed=3,transfer.repr_phase=Valid,exp_name="small_sweep_adult/data--adult--model_class-WeightedEqoppWassGan--model_fair_coeff-0_1/transfer_epoch_number-0--transfer_model_seed-3--transfer_repr_phase-Valid"
python src/run_unf_clf.py sweeps/small_sweep_adult/config.json --dirs local --data adult -o model.class=WeightedEqoppWassGan,model.fair_coeff=0.2,transfer.epoch_number=0,transfer.model_seed=1,transfer.repr_phase=Test,exp_name="small_sweep_adult/data--adult--model_class-WeightedEqoppWassGan--model_fair_coeff-0_2/transfer_epoch_number-0--transfer_model_seed-1--transfer_repr_phase-Test"
python src/run_unf_clf.py sweeps/small_sweep_adult/config.json --dirs local --data adult -o model.class=WeightedEqoppWassGan,model.fair_coeff=0.2,transfer.epoch_number=0,transfer.model_seed=1,transfer.repr_phase=Valid,exp_name="small_sweep_adult/data--adult--model_class-WeightedEqoppWassGan--model_fair_coeff-0_2/transfer_epoch_number-0--transfer_model_seed-1--transfer_repr_phase-Valid"
python src/run_unf_clf.py sweeps/small_sweep_adult/config.json --dirs local --data adult -o model.class=WeightedEqoppWassGan,model.fair_coeff=0.2,transfer.epoch_number=0,transfer.model_seed=2,transfer.repr_phase=Test,exp_name="small_sweep_adult/data--adult--model_class-WeightedEqoppWassGan--model_fair_coeff-0_2/transfer_epoch_number-0--transfer_model_seed-2--transfer_repr_phase-Test"
python src/run_unf_clf.py sweeps/small_sweep_adult/config.json --dirs local --data adult -o model.class=WeightedEqoppWassGan,model.fair_coeff=0.2,transfer.epoch_number=0,transfer.model_seed=2,transfer.repr_phase=Valid,exp_name="small_sweep_adult/data--adult--model_class-WeightedEqoppWassGan--model_fair_coeff-0_2/transfer_epoch_number-0--transfer_model_seed-2--transfer_repr_phase-Valid"
python src/run_unf_clf.py sweeps/small_sweep_adult/config.json --dirs local --data adult -o model.class=WeightedEqoppWassGan,model.fair_coeff=0.2,transfer.epoch_number=0,transfer.model_seed=3,transfer.repr_phase=Test,exp_name="small_sweep_adult/data--adult--model_class-WeightedEqoppWassGan--model_fair_coeff-0_2/transfer_epoch_number-0--transfer_model_seed-3--transfer_repr_phase-Test"
python src/run_unf_clf.py sweeps/small_sweep_adult/config.json --dirs local --data adult -o model.class=WeightedEqoppWassGan,model.fair_coeff=0.2,transfer.epoch_number=0,transfer.model_seed=3,transfer.repr_phase=Valid,exp_name="small_sweep_adult/data--adult--model_class-WeightedEqoppWassGan--model_fair_coeff-0_2/transfer_epoch_number-0--transfer_model_seed-3--transfer_repr_phase-Valid"
python src/run_unf_clf.py sweeps/small_sweep_adult/config.json --dirs local --data adult -o model.class=WeightedEqoppWassGan,model.fair_coeff=0.3,transfer.epoch_number=0,transfer.model_seed=1,transfer.repr_phase=Test,exp_name="small_sweep_adult/data--adult--model_class-WeightedEqoppWassGan--model_fair_coeff-0_3/transfer_epoch_number-0--transfer_model_seed-1--transfer_repr_phase-Test"
python src/run_unf_clf.py sweeps/small_sweep_adult/config.json --dirs local --data adult -o model.class=WeightedEqoppWassGan,model.fair_coeff=0.3,transfer.epoch_number=0,transfer.model_seed=1,transfer.repr_phase=Valid,exp_name="small_sweep_adult/data--adult--model_class-WeightedEqoppWassGan--model_fair_coeff-0_3/transfer_epoch_number-0--transfer_model_seed-1--transfer_repr_phase-Valid"
python src/run_unf_clf.py sweeps/small_sweep_adult/config.json --dirs local --data adult -o model.class=WeightedEqoppWassGan,model.fair_coeff=0.3,transfer.epoch_number=0,transfer.model_seed=2,transfer.repr_phase=Test,exp_name="small_sweep_adult/data--adult--model_class-WeightedEqoppWassGan--model_fair_coeff-0_3/transfer_epoch_number-0--transfer_model_seed-2--transfer_repr_phase-Test"
python src/run_unf_clf.py sweeps/small_sweep_adult/config.json --dirs local --data adult -o model.class=WeightedEqoppWassGan,model.fair_coeff=0.3,transfer.epoch_number=0,transfer.model_seed=2,transfer.repr_phase=Valid,exp_name="small_sweep_adult/data--adult--model_class-WeightedEqoppWassGan--model_fair_coeff-0_3/transfer_epoch_number-0--transfer_model_seed-2--transfer_repr_phase-Valid"
python src/run_unf_clf.py sweeps/small_sweep_adult/config.json --dirs local --data adult -o model.class=WeightedEqoppWassGan,model.fair_coeff=0.3,transfer.epoch_number=0,transfer.model_seed=3,transfer.repr_phase=Test,exp_name="small_sweep_adult/data--adult--model_class-WeightedEqoppWassGan--model_fair_coeff-0_3/transfer_epoch_number-0--transfer_model_seed-3--transfer_repr_phase-Test"
python src/run_unf_clf.py sweeps/small_sweep_adult/config.json --dirs local --data adult -o model.class=WeightedEqoppWassGan,model.fair_coeff=0.3,transfer.epoch_number=0,transfer.model_seed=3,transfer.repr_phase=Valid,exp_name="small_sweep_adult/data--adult--model_class-WeightedEqoppWassGan--model_fair_coeff-0_3/transfer_epoch_number-0--transfer_model_seed-3--transfer_repr_phase-Valid"
python src/run_unf_clf.py sweeps/small_sweep_adult/config.json --dirs local --data adult -o model.class=DemParGan,model.fair_coeff=0.1,transfer.epoch_number=0,transfer.model_seed=1,transfer.repr_phase=Test,exp_name="small_sweep_adult/data--adult--model_class-DemParGan--model_fair_coeff-0_1/transfer_epoch_number-0--transfer_model_seed-1--transfer_repr_phase-Test"
python src/run_unf_clf.py sweeps/small_sweep_adult/config.json --dirs local --data adult -o model.class=DemParGan,model.fair_coeff=0.1,transfer.epoch_number=0,transfer.model_seed=1,transfer.repr_phase=Valid,exp_name="small_sweep_adult/data--adult--model_class-DemParGan--model_fair_coeff-0_1/transfer_epoch_number-0--transfer_model_seed-1--transfer_repr_phase-Valid"
python src/run_unf_clf.py sweeps/small_sweep_adult/config.json --dirs local --data adult -o model.class=DemParGan,model.fair_coeff=0.1,transfer.epoch_number=0,transfer.model_seed=2,transfer.repr_phase=Test,exp_name="small_sweep_adult/data--adult--model_class-DemParGan--model_fair_coeff-0_1/transfer_epoch_number-0--transfer_model_seed-2--transfer_repr_phase-Test"
python src/run_unf_clf.py sweeps/small_sweep_adult/config.json --dirs local --data adult -o model.class=DemParGan,model.fair_coeff=0.1,transfer.epoch_number=0,transfer.model_seed=2,transfer.repr_phase=Valid,exp_name="small_sweep_adult/data--adult--model_class-DemParGan--model_fair_coeff-0_1/transfer_epoch_number-0--transfer_model_seed-2--transfer_repr_phase-Valid"
python src/run_unf_clf.py sweeps/small_sweep_adult/config.json --dirs local --data adult -o model.class=DemParGan,model.fair_coeff=0.1,transfer.epoch_number=0,transfer.model_seed=3,transfer.repr_phase=Test,exp_name="small_sweep_adult/data--adult--model_class-DemParGan--model_fair_coeff-0_1/transfer_epoch_number-0--transfer_model_seed-3--transfer_repr_phase-Test"
python src/run_unf_clf.py sweeps/small_sweep_adult/config.json --dirs local --data adult -o model.class=DemParGan,model.fair_coeff=0.1,transfer.epoch_number=0,transfer.model_seed=3,transfer.repr_phase=Valid,exp_name="small_sweep_adult/data--adult--model_class-DemParGan--model_fair_coeff-0_1/transfer_epoch_number-0--transfer_model_seed-3--transfer_repr_phase-Valid"
python src/run_unf_clf.py sweeps/small_sweep_adult/config.json --dirs local --data adult -o model.class=DemParGan,model.fair_coeff=0.2,transfer.epoch_number=0,transfer.model_seed=1,transfer.repr_phase=Test,exp_name="small_sweep_adult/data--adult--model_class-DemParGan--model_fair_coeff-0_2/transfer_epoch_number-0--transfer_model_seed-1--transfer_repr_phase-Test"
python src/run_unf_clf.py sweeps/small_sweep_adult/config.json --dirs local --data adult -o model.class=DemParGan,model.fair_coeff=0.2,transfer.epoch_number=0,transfer.model_seed=1,transfer.repr_phase=Valid,exp_name="small_sweep_adult/data--adult--model_class-DemParGan--model_fair_coeff-0_2/transfer_epoch_number-0--transfer_model_seed-1--transfer_repr_phase-Valid"
python src/run_unf_clf.py sweeps/small_sweep_adult/config.json --dirs local --data adult -o model.class=DemParGan,model.fair_coeff=0.2,transfer.epoch_number=0,transfer.model_seed=2,transfer.repr_phase=Test,exp_name="small_sweep_adult/data--adult--model_class-DemParGan--model_fair_coeff-0_2/transfer_epoch_number-0--transfer_model_seed-2--transfer_repr_phase-Test"
python src/run_unf_clf.py sweeps/small_sweep_adult/config.json --dirs local --data adult -o model.class=DemParGan,model.fair_coeff=0.2,transfer.epoch_number=0,transfer.model_seed=2,transfer.repr_phase=Valid,exp_name="small_sweep_adult/data--adult--model_class-DemParGan--model_fair_coeff-0_2/transfer_epoch_number-0--transfer_model_seed-2--transfer_repr_phase-Valid"
python src/run_unf_clf.py sweeps/small_sweep_adult/config.json --dirs local --data adult -o model.class=DemParGan,model.fair_coeff=0.2,transfer.epoch_number=0,transfer.model_seed=3,transfer.repr_phase=Test,exp_name="small_sweep_adult/data--adult--model_class-DemParGan--model_fair_coeff-0_2/transfer_epoch_number-0--transfer_model_seed-3--transfer_repr_phase-Test"
python src/run_unf_clf.py sweeps/small_sweep_adult/config.json --dirs local --data adult -o model.class=DemParGan,model.fair_coeff=0.2,transfer.epoch_number=0,transfer.model_seed=3,transfer.repr_phase=Valid,exp_name="small_sweep_adult/data--adult--model_class-DemParGan--model_fair_coeff-0_2/transfer_epoch_number-0--transfer_model_seed-3--transfer_repr_phase-Valid"
python src/run_unf_clf.py sweeps/small_sweep_adult/config.json --dirs local --data adult -o model.class=DemParGan,model.fair_coeff=0.3,transfer.epoch_number=0,transfer.model_seed=1,transfer.repr_phase=Test,exp_name="small_sweep_adult/data--adult--model_class-DemParGan--model_fair_coeff-0_3/transfer_epoch_number-0--transfer_model_seed-1--transfer_repr_phase-Test"
python src/run_unf_clf.py sweeps/small_sweep_adult/config.json --dirs local --data adult -o model.class=DemParGan,model.fair_coeff=0.3,transfer.epoch_number=0,transfer.model_seed=1,transfer.repr_phase=Valid,exp_name="small_sweep_adult/data--adult--model_class-DemParGan--model_fair_coeff-0_3/transfer_epoch_number-0--transfer_model_seed-1--transfer_repr_phase-Valid"
python src/run_unf_clf.py sweeps/small_sweep_adult/config.json --dirs local --data adult -o model.class=DemParGan,model.fair_coeff=0.3,transfer.epoch_number=0,transfer.model_seed=2,transfer.repr_phase=Test,exp_name="small_sweep_adult/data--adult--model_class-DemParGan--model_fair_coeff-0_3/transfer_epoch_number-0--transfer_model_seed-2--transfer_repr_phase-Test"
python src/run_unf_clf.py sweeps/small_sweep_adult/config.json --dirs local --data adult -o model.class=DemParGan,model.fair_coeff=0.3,transfer.epoch_number=0,transfer.model_seed=2,transfer.repr_phase=Valid,exp_name="small_sweep_adult/data--adult--model_class-DemParGan--model_fair_coeff-0_3/transfer_epoch_number-0--transfer_model_seed-2--transfer_repr_phase-Valid"
python src/run_unf_clf.py sweeps/small_sweep_adult/config.json --dirs local --data adult -o model.class=DemParGan,model.fair_coeff=0.3,transfer.epoch_number=0,transfer.model_seed=3,transfer.repr_phase=Test,exp_name="small_sweep_adult/data--adult--model_class-DemParGan--model_fair_coeff-0_3/transfer_epoch_number-0--transfer_model_seed-3--transfer_repr_phase-Test"
python src/run_unf_clf.py sweeps/small_sweep_adult/config.json --dirs local --data adult -o model.class=DemParGan,model.fair_coeff=0.3,transfer.epoch_number=0,transfer.model_seed=3,transfer.repr_phase=Valid,exp_name="small_sweep_adult/data--adult--model_class-DemParGan--model_fair_coeff-0_3/transfer_epoch_number-0--transfer_model_seed-3--transfer_repr_phase-Valid"
