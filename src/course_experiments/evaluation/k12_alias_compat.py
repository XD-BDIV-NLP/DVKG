from __future__ import annotations


def entity_names(retriever,nid):
    entity=retriever.entities[nid]
    aliases=entity.get('aliases',[])
    if aliases is None:aliases=[]
    return sorted({value for value in [entity.get('canonical_name',entity.get('name','')),*aliases]
                   if isinstance(value,str) and value})


def compatible_worker_command(command,launcher):
    recognized=('course_experiments.evaluation.run_repaired_full' in command
                or any(str(part).endswith('supplement_fair_runner.py') for part in command))
    if recognized and '--job' in command and '--base' in command:
        name=command[command.index('--job')+1]
        if name in {'osAB_K12','supp_osAB_K12'}:
            return [command[0],'-u',str(launcher),'worker',*command[command.index('--base'):]]
    return command
