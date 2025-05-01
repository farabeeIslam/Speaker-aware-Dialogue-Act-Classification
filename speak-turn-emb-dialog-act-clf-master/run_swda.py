import os

# Fast training configuration
if __name__ == '__main__':
    corpus = 'swda'
    mode = ['train', 'inference'][0]  # train mode
    batch_size = 32                  # 🟰 increased batch size
    batch_size_val = 32               # 🟰 validation batch size
    emb_batch = 0                     # for special embedding batching, keep 0
    epochs = 1                        # 🟰 only 1 epoch to finish fast
    gpu = '0'                         # 🟰 use only 1 GPU (gpu id 0)
    lr = 1e-4                         # learning rate
    nlayer = 2                        # number of GRU layers
    chunk_size = 128                  # chunk size for long conversations
    dropout = 0.5                     # dropout rate
    nfinetune = 0                     # 🟰 no BERT layers finetuned (fully frozen)
    nclass = 43                       # number of dialogue act classes
    speaker_info = 'emb_cls'           # use speaker embeddings
    topic_info = 'none'                # do not use topic embeddings

    # Make result directory if it doesn't exist
    os.makedirs(f'results_{corpus}', exist_ok=True)

    # Build the result file name
    file_name = f"results_{corpus}/{corpus}_chunk={chunk_size}_nlayer={nlayer}.txt"

    if nfinetune != 2:
        file_name = file_name[:-4] + f'_nfinetune={nfinetune}.txt'

    if speaker_info != 'none':
        file_name = file_name[:-4] + f'_speaker={speaker_info}.txt'

    if topic_info != 'none':
        file_name = file_name[:-4] + f'_tinfo={topic_info}.txt'

    if lr != 1e-4:
        file_name = file_name[:-4] + f'_lr={lr}.txt'

    if dropout != 0.5:
        file_name = file_name[:-4] + f'_dropout={dropout}.txt'

    if not gpu:
        n_gpu = 0
    else:
        n_gpu = len(gpu.split(','))
    if n_gpu != 2:
        file_name = file_name[:-4] + f'_ngpu={n_gpu}.txt'

    # Create the command string to run engine.py
    command = f"python -u engine.py --corpus={corpus} --mode={mode} --gpu={gpu} " \
              f"--batch_size={batch_size} --batch_size_val={batch_size_val} " \
              f"--epochs={epochs} --lr={lr} --nlayer={nlayer} --chunk_size={chunk_size} " \
              f"--dropout={dropout} --nfinetune={nfinetune} --speaker_info={speaker_info} " \
              f"--topic_info={topic_info} --nclass={nclass} --emb_batch={emb_batch}"

    print(command)  # show the command that will be executed
    os.system(command)  # execute the command