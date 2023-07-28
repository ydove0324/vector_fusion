import subprocess
import argparse
import sys
# parser = argparse.ArgumentParser(description='test')
# parser.add_argument('--env',type = str,default='base')
# parser.add_argument('--dir',type = str)


def run_command(str_list,index):
    print(str_list)
    result = subprocess.run(str_list,capture_output=True,text=True)
    if result.returncode != 0:
        raise Exception("#error occured:\n",result.stderr)
    else:
        index += 1
        print("{} succeed:\n".format(index),result.stdout)
    
def activate_env(env,index):
    run_command(['conda','init'],index)
    command_list = ['conda','activate',env]
    run_command(command_list,index)
def cd(dir,index):
    command_list = ['cd',dir]
    run_command(command_list,index)

# args = parser.parse_args()

index = 0

# if args.env is not None:
#     activate_env(args.env,index)
# if args.dir is not None:
#     cd(args.dir,index)
run_command(['where.exe','python'],index)
str_list = sys.stdin.read().split('\n')
str_list.pop()
print(str_list)
for str in str_list:
    print(str)
    run_command(str.split(' '),index)


