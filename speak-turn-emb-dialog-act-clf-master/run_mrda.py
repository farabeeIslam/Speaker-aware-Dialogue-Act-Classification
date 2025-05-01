import os

# Fast training configuration for MRDA
if __name__ == '__main__':
    corpus = 'mrda'
    mode = ['train', 'inference'][0]  # Choose 'train'
    batch_size =2                   # Small to prevent OOM
    batch_size_val =2
    emb_batch = 0
    epochs = 1                       #   Only 1 epoch for speed
    gpu = '0'                        # Use only GPU 0
    lr = 1e-4
    nlayer = 2
    chunk_size = 64
    dropout = 0.5
    nfinetune = 0
    nclass = 5                      # MRDA has 5 classes
    speaker_info = 'emb_cls'
    topic_info = 'none'

    os.makedirs(f'results_{corpus}', exist_ok=True)

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

    command = f"python -u engine.py --corpus={corpus} --mode={mode} --gpu={gpu} " \
              f"--batch_size={batch_size} --batch_size_val={batch_size_val} " \
              f"--epochs={epochs} --lr={lr} --nlayer={nlayer} --chunk_size={chunk_size} " \
              f"--dropout={dropout} --nfinetune={nfinetune} --speaker_info={speaker_info} " \
              f"--topic_info={topic_info} --nclass={nclass} --emb_batch={emb_batch}"

    print(command)
    os.system(command + f" >> {file_name}")