# strands-decider-docker

A container image that serves [Strands Decider](https://github.com/strands-labs/strands-decider)
over HTTP with all model weights baked in, so it runs on machines that cannot reach Hugging Face.

GitHub Actions builds the image and pushes it to `ghcr.io/ninjakaib/strands-decider`
(`.github/workflows/build-image.yml`). Run the workflow by hand to pick a different decider
checkpoint or revision.

## Run

Needs an NVIDIA GPU and the NVIDIA Container Toolkit.

```bash
docker run --rm --gpus all -p 127.0.0.1:8000:8000 ghcr.io/ninjakaib/strands-decider:latest
curl -s localhost:8000/health
curl -s localhost:8000/v1/systemone -H 'content-type: application/json' -d '{
  "state": "Help! My payouts have been failing for 3 days.",
  "questions": {"is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"}}
}'
```

The server has no authentication, so publish the port on 127.0.0.1 only.
