# GitHub Pages: review before merging

The article and browser deck are one static site. This branch adds the files
under `site/` and the Pages workflow under `.github/workflows/pages.yml`.

## Share this branch for feedback

1. Push the `docs/visual-explainer` branch and open a draft pull request.
2. A repository administrator enables **Settings → Pages → Build and deployment
   → Source → GitHub Actions**.
3. In **Settings → Environments → github-pages**, permit deployments from
   `main` and `docs/visual-explainer`. Keep any required-reviewer rules.
4. The branch push runs **Illustrated guide and review preview**. If Pages was
   enabled after the first run, rerun the workflow from the Actions page.
5. Once deployment succeeds, share:

   - Article: <https://meta-pytorch.github.io/dist_moe/previews/visual-explainer/>
   - Deck: <https://meta-pytorch.github.io/dist_moe/previews/visual-explainer/?view=slides>

GitHub Pages provides one deployment per repository, rather than automatic
public PR-preview URLs. This workflow puts the review branch in its own
`previews/visual-explainer/` directory and assembles the production root from
`main`. Updating the review branch updates that preview without replacing the
main-branch site.

Before `main` contains the site, the root links/redirects to the review preview.
After merging, the production article is at
<https://meta-pytorch.github.io/dist_moe/> and its deck at
<https://meta-pytorch.github.io/dist_moe/?view=slides>.

If the review branch is later deleted, a subsequent main deployment omits the
preview and keeps the production site.

## Review artifacts without enabling Pages

The workflow's assembly job also uploads a `github-pages` artifact. GitHub users
with repository access can download it from the workflow run, extract
`artifact.tar`, and serve it locally. That is a downloadable review artifact;
a shareable browser URL becomes available after Pages deployment succeeds.

## Local deployment check

From the repository root, choose a new output directory:

```sh
python .github/scripts/assemble_pages.py \
  --preview site --output /tmp/dist-moe-pages-preview
python -m http.server 8000 --bind 127.0.0.1 \
  --directory /tmp/dist-moe-pages-preview
```

Open `/previews/visual-explainer/` for the article and append `?view=slides`
for the deck. The assembly script checks local asset references. The content
runs in the browser and needs no CUDA, Node package installation, or build step.
