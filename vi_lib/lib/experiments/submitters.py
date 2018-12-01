# @Author: amishkin
# @Date:   18-09-07
# @Email:  amishkin@cs.ubc.ca
# @Last modified by:   aaronmishkin
# @Last modified time: 18-09-07

import subprocess

def submit_python_jobs(name, variants, method='BBB', cv=0):
    subprocess.call("pwd", shell=True)
    for i, variant in enumerate(variants):
        command = "python ./lib/experiments/run_experiment.py --name=\"" + str(name) + "\" --variant=\"" + str(variant) + "\" --method=\"" + str(method) + "\" --cv=" + str(cv)
        exit_status = subprocess.call(command, shell=True)
        if exit_status == 1:
            print("Job {0} failed to submit".format(command))
    print("Done submitting jobs!")

def profiled_submit_python_jobs(name, variants, method='BBB', cv=0, output='local.cprof'):
    subprocess.call("pwd", shell=True)
    for i, variant in enumerate(variants):
    #        command = "python -m torch.utils.bottleneck ./lib/experiments/run_experiment.py --name=\"" + str(name) + "\" --variant=\"" + str(variant) + "\" --method=\"" + str(method) + "\" --cv=" + str(cv) + " > " + str(name) + str(variant) + ".cprof"
        command = "python -m cProfile -o " + output + " ./lib/experiments/run_experiment.py --name=\"" + str(name) + "\" --variant=\"" + str(variant) + "\" --method=\"" + str(method) + "\" --cv=" + str(cv)
    
        exit_status = subprocess.call(command, shell=True)
        if exit_status == 1:
            print("Job {0} failed to submit".format(command))
    print("Done submitting jobs!")

def submit_raiden_jobs(name, variants, method='BBB', cv=0):
    subprocess.call("pwd", shell=True)
    for i, variant in enumerate(variants):
        qsub_com = "qsub -ac d=nvcr-digits-1611 -jc gpu-container_g1.24h -v METHOD=" + method + ",NAME="+str(name)+",VARIANT="+str(variant)+",CV="+str(cv)+",SCRIPT="+str("./lib/experiments/run_experiment.py")+" "+str("qsub_execute.sh")+""
        exit_status = subprocess.call(qsub_com, shell=True)
        if exit_status == 1:
            print("Job {0} failed to submit".format(qsub_com))
    print("Done submitting jobs!")

def submit_raiden_jobs_cpu_normal(name, variants, method='BBB', cv=0):
    subprocess.call("pwd", shell=True)
    for i, variant in enumerate(variants):
        qsub_com = "qsub -jc pcc-normal.24h -v METHOD=" + method + ",NAME="+str(name)+",VARIANT="+str(variant)+",CV="+str(cv)+",SCRIPT="+str("./lib/experiments/run_experiment.py")+" "+str("qsub_execute.sh")+""
        exit_status = subprocess.call(qsub_com, shell=True)
        if exit_status == 1:
            print("Job {0} failed to submit".format(qsub_com))
    print("Done submitting jobs!")

def submit_raiden_jobs_cpu_large(name, variants, method='BBB', cv=0):
    subprocess.call("pwd", shell=True)
    for i, variant in enumerate(variants):
        qsub_com = "qsub -jc pcc-large.24h -v METHOD=" + method + ",NAME="+str(name)+",VARIANT="+str(variant)+",CV="+str(cv)+",SCRIPT="+str("./lib/experiments/run_experiment.py")+" "+str("qsub_execute.sh")+""
        exit_status = subprocess.call(qsub_com, shell=True)
        if exit_status == 1:
            print("Job {0} failed to submit".format(qsub_com))
    print("Done submitting jobs!")

def submit_raiden_jobs_cpu_skl(name, variants, method='BBB', cv=0):
    subprocess.call("pwd", shell=True)
    for i, variant in enumerate(variants):
        qsub_com = "qsub -jc pcc-skl.24h -v METHOD=" + method + ",NAME="+str(name)+",VARIANT="+str(variant)+",CV="+str(cv)+",SCRIPT="+str("./lib/experiments/run_experiment.py")+" "+str("qsub_execute.sh")+""
        exit_status = subprocess.call(qsub_com, shell=True)
        if exit_status == 1:
            print("Job {0} failed to submit".format(qsub_com))
    print("Done submitting jobs!")

def profiled_submit_raiden_jobs(name, variants, method='BBB', cv=0, output='gpu.cprof'):
    subprocess.call("pwd", shell=True)
    for i, variant in enumerate(variants):
        qsub_com = "qsub -ac d=nvcr-digits-1611 -jc gpu-container_g1.24h -v METHOD=" + method + ",NAME="+str(name)+",VARIANT="+str(variant)+",CV="+str(cv)+",OUTPUT="+str(output)+",SCRIPT="+str("./lib/experiments/run_experiment.py")+" "+str("profiled_qsub_execute.sh")+""
        exit_status = subprocess.call(qsub_com, shell=True)
        if exit_status == 1:
            print("Job {0} failed to submit".format(qsub_com))
    print("Done submitting jobs!")

def profiled_submit_raiden_jobs_cpu_normal(name, variants, method='BBB', cv=0, output='cpu_normal.cprof'):
    subprocess.call("pwd", shell=True)
    for i, variant in enumerate(variants):
        qsub_com = "qsub -jc pcc-normal.24h -v METHOD=" + method + ",NAME="+str(name)+",VARIANT="+str(variant)+",CV="+str(cv)+",OUTPUT="+str(output)+",SCRIPT="+str("./lib/experiments/run_experiment.py")+" "+str("profiled_qsub_execute.sh")+""
        exit_status = subprocess.call(qsub_com, shell=True)
        if exit_status == 1:
            print("Job {0} failed to submit".format(qsub_com))
    print("Done submitting jobs!")

def profiled_submit_raiden_jobs_cpu_large(name, variants, method='BBB', cv=0, output='cpu_large.cprof'):
    subprocess.call("pwd", shell=True)
    for i, variant in enumerate(variants):
        qsub_com = "qsub -jc pcc-large.24h -v METHOD=" + method + ",NAME="+str(name)+",VARIANT="+str(variant)+",CV="+str(cv)+",OUTPUT="+str(output)+",SCRIPT="+str("./lib/experiments/run_experiment.py")+" "+str("profiled_qsub_execute.sh")+""
        exit_status = subprocess.call(qsub_com, shell=True)
        if exit_status == 1:
            print("Job {0} failed to submit".format(qsub_com))
    print("Done submitting jobs!")

def profiled_submit_raiden_jobs_cpu_skl(name, variants, method='BBB', cv=0, output='cpu_skl.cprof'):
    subprocess.call("pwd", shell=True)
    for i, variant in enumerate(variants):
        qsub_com = "qsub -jc pcc-skl.24h -v METHOD=" + method + ",NAME="+str(name)+",VARIANT="+str(variant)+",CV="+str(cv)+",OUTPUT="+str(output)+",SCRIPT="+str("./lib/experiments/run_experiment.py")+" "+str("profiled_qsub_execute.sh")+""
        exit_status = subprocess.call(qsub_com, shell=True)
        if exit_status == 1:
            print("Job {0} failed to submit".format(qsub_com))
    print("Done submitting jobs!")
