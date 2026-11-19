"""MLflow over its REST API with the machine's own settings (MLFLOW_TRACKING_URI, _USERNAME, _PASSWORD), nothing printed of them."""
import base64, json, os, sys, urllib.error, urllib.parse, urllib.request

BASE = os.environ["MLFLOW_TRACKING_URI"].rstrip("/")
_auth = base64.b64encode(("%s:%s" % (os.environ.get("MLFLOW_TRACKING_USERNAME", ""), os.environ.get("MLFLOW_TRACKING_PASSWORD", ""))).encode()).decode()


def call(method, path, body=None, query=None):
    url = BASE + "/api/2.0/mlflow/" + path + ("?" + urllib.parse.urlencode(query) if query else "")
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={"Authorization": "Basic " + _auth, "Content-Type": "application/json", "User-Agent": "mlflow-relay/1.0", "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            text = r.read()
            return json.loads(text) if text else {}
    except urllib.error.HTTPError as e:
        raise SystemExit("mlflow %s %s: %d %s" % (method, path, e.code, e.read()[:400].decode("utf-8", "replace")))


def experiment(name):
    try:
        return call("GET", "experiments/get-by-name", query={"experiment_name": name})["experiment"]["experiment_id"]
    except SystemExit as e:
        if "RESOURCE_DOES_NOT_EXIST" not in str(e):
            raise
    return call("POST", "experiments/create", {"name": name})["experiment_id"]


if __name__ == "__main__":
    if sys.argv[1] == "list":
        for e in call("POST", "experiments/search", {"max_results": 100}).get("experiments", []):
            print(e["experiment_id"], e["name"], e.get("lifecycle_stage"))
    elif sys.argv[1] == "make":
        print(experiment(sys.argv[2]))
