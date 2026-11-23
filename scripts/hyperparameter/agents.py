#!/usr/bin/env python3
# agents.py PORT N PREFIX: N sessions that each take TURNS turns on their own slot, as an agent does: a new prompt
# of 60 to 200 tokens (cache off, so it is read whole, as a tool result is) and a reply of 12 to 40 tokens, the next
# turn sent as soon as the last one returns; session s starts s*ARRIVE_MS after the first. Prompts therefore arrive
# while other sessions reply. Prints the pass's reply tokens a second in all and a session's mean rate while replying.
import json, os, random, sys, threading, time, urllib.request
port, n, prefix = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
turns = int(os.environ.get('TURNS', '6')); arrive = int(os.environ.get('ARRIVE_MS', '150'))
res = [[] for _ in range(n)]
def session(s):
    time.sleep(s*arrive/1000)
    r = random.Random(1000 + s)
    for t in range(turns):
        prompt = [r.randrange(3, 150000) for _ in range(r.randrange(60, 201))]
        body = {'prompt': prompt, 'n_predict': r.randrange(12, 41), 'temperature': 1.0, 'top_p': 0.95, 'top_k': 40,
                'repeat_penalty': 1.1, 'repeat_last_n': 256, 'seed': 530 + s, 'cache_prompt': False, 'id_slot': s}
        req = urllib.request.Request('http://localhost:%d/completion' % port, json.dumps(body).encode(), {'Content-Type': 'application/json'})
        res[s].append(json.load(urllib.request.urlopen(req, timeout=600))['timings'])
t0 = time.time()
th = [threading.Thread(target=session, args=(s,)) for s in range(n)]
for x in th: x.start()
for x in th: x.join()
wall = time.time() - t0
tim = [x for r in res for x in r]
pred = sum(x['predicted_n'] for x in tim); prom = sum(x['prompt_n'] for x in tim)
json.dump(res, open(prefix + '-agents.json', 'w'))
print('agents: %d sessions x %d turns in %.2f s: %d reply tokens, %.1f tok/s in all; %d prompt tokens; a reply ran at %.1f tok/s (mean of %d), prompts at %.0f tok/s'
      % (n, turns, wall, pred, pred/wall, prom, sum(x['predicted_per_second'] for x in tim)/len(tim), len(tim), sum(x['prompt_per_second'] for x in tim)/len(tim)))
