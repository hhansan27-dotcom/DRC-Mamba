# Uploading this repository with GitHub Desktop

The `pretrained/` files are configured for Git LFS because each checkpoint is about 89 MB.
This keeps the normal Git history lightweight and avoids repeated large binary objects.

> **Checkpoint / Git LFS note**
>
> This release ZIP already contains the three real pretrained checkpoint binaries (about
> 89 MB each), not LFS pointer placeholders. The included `.gitattributes` tracks
> `pretrained/*.pth` with Git LFS. Install Git LFS before the first commit, then add and
> push the repository normally; Git LFS will upload the checkpoint objects.

## 1. Install / initialize Git LFS once

Open PowerShell or Git Bash and run:

```bash
git lfs install
```

If `git lfs` is not recognized, install Git LFS from https://git-lfs.com/ and run the command again.
GitHub Desktop works with repositories that use Git LFS.

## 2. Extract the project

Extract the final ZIP. Open the **DRC-Mamba** folder itself, not the ZIP file.
The folder containing `README.md`, `train.py`, `drc_mamba/`, and `pretrained/` is the repository root.

## 3. Add it to GitHub Desktop

1. Open GitHub Desktop and sign in to the GitHub account `hhansan27-dotcom`.
2. Choose **File -> Add Local Repository...**.
3. Select the extracted `DRC-Mamba` folder.
4. If GitHub Desktop says it is not a Git repository, choose **create a repository here**.
5. Repository name: `DRC-Mamba`.
6. Leave the local path unchanged. Do not initialize a second README or `.gitignore`.

## 4. Make the first commit

In the **Changes** panel, verify that source files and the three files under `pretrained/` are listed.
The weight files should be handled by Git LFS because `.gitattributes` is already included.

Use a summary such as:

```text
Initial release of DRC-Mamba
```

Then click **Commit to main**.

## 5. Publish to GitHub

Click **Publish repository**.

Recommended settings:

- Name: `DRC-Mamba`
- Owner: `hhansan27-dotcom`
- Description: `Official implementation of DRC-Mamba for infrared small target detection.`
- Keep the repository private during submission if desired; switch to public when ready.

After publishing, the repository URL will be:

```text
https://github.com/hhansan27-dotcom/DRC-Mamba
```

## 6. Verify the upload

On GitHub, confirm that:

- `README.md` renders on the front page;
- `drc_mamba/`, `scripts/`, `tests/`, and `pretrained/` are present;
- each pretrained `.pth` page shows that it is stored with Git LFS;
- the ZIP file itself was **not** committed.

## Future updates

After editing files, GitHub Desktop only needs:

1. review **Changes**;
2. enter a commit message;
3. **Commit to main**;
4. **Push origin**.
