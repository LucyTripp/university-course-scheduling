# Publish to GitHub

Suggested repository name: `university-course-scheduling`.

Suggested description: `University course timetabling with OR-Tools CP-SAT: comparing classroom efficiency and student scheduling conflicts.`

1. Extract the project ZIP and open its project folder in VS Code.
2. Create an empty repository on GitHub with the suggested name and your chosen visibility. Do not initialize it with a README or license; this package already contains a README.
3. Run the following in the project folder, replacing YOUR_USERNAME with your GitHub username:

```powershell
git init -b main
git add .
git status
```

Check the staged list: data XML, processed CSV inputs, virtual environments, and `outputs/` should not appear. Code, documentation, tests, and `results/` should appear.

```powershell
git commit -m "Add university course scheduling project"
git remote add origin https://github.com/YOUR_USERNAME/university-course-scheduling.git
git push -u origin main
```

Use your existing GitHub authentication when prompted. Never put credentials in repository files. Do not use GitHub's web uploader to upload the whole ZIP: Git's ignore rules only protect the input files when publishing with Git.

Before making the repository public, choose a code license if you want to allow reuse. MIT is a common permissive choice, but it must not be applied to the third-party dataset. The README links to the exact verified source download so readers can regenerate the data.
