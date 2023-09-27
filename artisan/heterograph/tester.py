import subprocess
import os

def metaprog_engine(instance):
    source = instance.defs['source']
    expect_retcode = instance.defs.get('retcode', 0)

    env_vars = os.environ.copy()

    defs = { k:str(v) for (k,v) in instance.defs.items() }

    env_vars.update(defs)

    return_code = subprocess.call(['artisan', source], env=env_vars)

    assert expect_retcode == return_code

