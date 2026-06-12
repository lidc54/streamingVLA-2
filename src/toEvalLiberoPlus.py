import subprocess
import os, sys
import json
import logging, time
from collections import defaultdict
from parse_plus_result import parse_log_folder

def build_suite_category():
    from libero.libero.benchmark import libero_suite_task_map
    task_cls = os.path.join(os.path.dirname(libero_suite_task_map.__file__), "task_classification.json")
    with open(task_cls, "r", encoding="utf-8") as f: data = json.load(f)
    allcat2ids = {}
    for suite, cls_lst in data.items():
        cat2ids = defaultdict(list)
        for it in cls_lst: cat2ids[it["category"]].append(it["id"]-1)
        allcat2ids[suite] = cat2ids
    return allcat2ids

def file_to_string(filename):
    with open(filename, 'r') as file:
        return file.read()

def set_freest_gpu():
    freest_gpu=0
    while True:
        freest_gpu, free_memory = get_freest_gpu()
        if free_memory < 5000:# 7G memory
            time.sleep(50)# 10 second
        else:
            break
    os.environ['CUDA_VISIBLE_DEVICES'] = str(freest_gpu)
    os.environ['HF_HUB_OFFLINE'] = str(1)

def get_freest_gpu():
    sp = subprocess.Popen(['gpustat', '--json'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out_str, _ = sp.communicate()
    gpustats = json.loads(out_str.decode('utf-8'))
    # Find GPU with most free memory
    freest_gpu = min(gpustats['gpus'], key=lambda x: x['memory.used'])
    free_memory = freest_gpu['memory.total'] - freest_gpu['memory.used']
    return freest_gpu['index'], free_memory


def block_until_testing(rl_filepath, log_status=False):
    # Ensure that the RL training has started before moving on
    while True:
        rl_log = file_to_string(rl_filepath)
        if "Success:" in rl_log or "Gym" in rl_log:
            if log_status:
                logging.info(f"successfully testing!")
            # if log_status and "fps step:" in rl_log:
            #     logging.info(f"Iteration {iter_num}: Code Run {response_id} successfully training!")
            # if log_status and "Traceback" in rl_log:
            #     logging.info(f"Iteration {iter_num}: Code Run {response_id} execution error!")
            break
    time.sleep(60)

def add_env_path(root):
    current_pythonpath = os.environ.get('PYTHONPATH', '')
    new_pythonpath = root + os.pathsep + current_pythonpath
    os.environ['PYTHONPATH'] = new_pythonpath.strip(os.pathsep)  # 去掉首尾多余的分隔符


def to_eval_one_by_one(files):
    eval_runs = []
    root = '/home/ldc/cc/home/lerobot_032/src'
    # root = os.path.dirname(__file__)
    add_env_path(root)

    allcat2ids=build_suite_category()
    output = 'outputs/train'
    # files = files.split()
    for job_name in files:
        job_name=job_name.strip()
        ecoch=100000 if 'param' not in job_name else 150000
        data_types=[]
        completed, incomplete = parse_log_folder(f'{output}/{job_name}/checkpoints/{ecoch}/pretrained_model')
        print("\n✅ 完成评估的结果:")
        print(json.dumps(completed, indent=4, ensure_ascii=False))
        print("\n⚠️ 未完成评估的文件 (按 nstep 分组):")
        print(json.dumps(incomplete, indent=4, ensure_ascii=False))
        
        cfg_file = f'{output}/{job_name}/checkpoints/{ecoch}/pretrained_model/config.json'
        tri_cfg_file = f'{output}/{job_name}/checkpoints/{ecoch}/pretrained_model/train_config.json'
        if not(os.path.exists(cfg_file) and os.path.exists(tri_cfg_file)):
            continue

        with open(tri_cfg_file, 'r', encoding='utf-8') as file:
            data = json.load(file)
            repo_id = data["dataset"]["repo_id"]
            if 'goal' in repo_id: data_types=['goal']
            if 'object' in repo_id: data_types=['object']
            if 'spatial' in repo_id: data_types=['spatial']
            if '10' in repo_id: data_types=['10']
            if not len(data_types): data_types=['10', 'goal', 'object', 'spatial']
            cuk=data['policy']['chunk_size']
        data = json.load(open(cfg_file, 'r', encoding='utf-8'))
        # all_tasks=['Background Textures', 'Robot Initial States', 'Camera Viewpoints', 'Language Instructions', 'Sensor Noise', 'Objects Layout', 'Light Conditions']
        for nas in [50, 10]:
            if nas > cuk: continue
            for data_type in data_types:
                data["n_action_steps"]=nas
                with open(cfg_file, 'w', encoding='utf-8') as f: json.dump(data, f, ensure_ascii=False, indent=4)
                # ['libero_spatial', 'libero_object', 'libero_goal', 'libero_10']
                for task_name, tasks_no_in_suite in allcat2ids[f'libero_{data_type}'].items():
                    if task_name not in incomplete.get(nas, {}).get(data_type, []): continue
                    print(f'\ntask_name: {task_name}, tasks_no_in_suite length: {len(tasks_no_in_suite)},from {tasks_no_in_suite[0]} to {tasks_no_in_suite[-1]}')
                    # Find the freest GPU to run GPU-accelerated RL
                    set_freest_gpu()
                    # Execute the python file with flags
                    rl_filepath = f"{output}/{job_name}/checkpoints/100000/pretrained_model/eval_{data_type}_nstep{nas}_{task_name.replace(' ','_')}.log"
                    with open(rl_filepath, 'w') as f:
                        process = subprocess.Popen(['python', '-u', 'lerobot/scripts/eval_LiberoPlus.py',
                                                    # 'hydra/output=subprocess',
                                                    f'--policy_path={output}/{job_name}/checkpoints/100000/pretrained_model/',
                                                    f'--task_suite_name=libero_{data_type}',
                                                    f'--tasks_no_in_suite={tasks_no_in_suite}',
                                                    f'--video_out_path={output}/{job_name}/checkpoints/100000/pretrained_model/video_eval_{data_type}_nstep{nas}'],
                                                    stdout=f, stderr=f)
                    block_until_testing(rl_filepath, log_status=True)
                    eval_runs.append(process)
                    print(f'job_name: {job_name}\nn_steps:{nas}; data_type: {data_type}; task_name: {task_name}\n\n')

    for i, rl_run in enumerate(eval_runs):
        rl_run.communicate(input='N\n')


if __name__ == "__main__":
    files=sys.argv[1:]
    to_eval_one_by_one(files)