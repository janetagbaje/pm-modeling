import optuna
from s_parser import parse_args
import train


def objective(trial):
    # Load the base config file
    data_config, encoder_config, decoder_config = parse_args()
    
    # Training and data parameters
    data_config['lr'] = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    data_config['batch_size'] = trial.suggest_int('batch_size', 16, 128)
    data_config['space_bands'] = trial.suggest_int('space_bands', 12, 128)
    data_config['num_sensors'] = trial.suggest_int('num_sensors', 6, 32)
    data_config['nlags'] = trial.suggest_int('nlags', 2, 25)

    # Encoder parameters
    encoder_config['enc_num_latent_channels'] = trial.suggest_int('enc_num_latent_channels', 4, 64)*encoder_config['num_cross_attention_heads']
    # encoder_config['enc_preproc_ch'] = trial.suggest_int('enc_preproc_ch', 8, 512)
    # decoder parameters
    decoder_config['dec_num_latent_channels'] = encoder_config['enc_num_latent_channels']
    #decoder_config['dec_preproc_ch'] = trial.suggest_int('dec_preproc_ch', 8, 512)

    print(encoder_config) 
    print(decoder_config) 
    return train.main(data_config, encoder_config, decoder_config)

if __name__ == "__main__":
    # Create an Optuna study
    study = optuna.create_study(direction="minimize")

    # Run optimization
    study.optimize(objective, n_trials=5)

    # Print the best trial
    best_trial = study.best_trial
    print(f"Best trial value: {best_trial.value}")
    print(f"Best hyperparameters: {best_trial.params}")