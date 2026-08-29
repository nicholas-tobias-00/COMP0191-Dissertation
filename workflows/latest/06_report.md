# 6. Dissertation build

The report is split by chapter, with appendices under `report/Appendices/`. The compiled artifact is
`report/Report.pdf`.

From `report/`, run two passes after ordinary text/reference changes:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error Report.tex
pdflatex -interaction=nonstopmode -halt-on-error Report.tex
```

Run BibTeX between LaTeX passes only when bibliography records or citation keys change:

```powershell
bibtex Report
```

Before treating the build as complete, check `Report.log` for undefined citations/references and
inspect any new overfull boxes introduced by the edited chapter.

Key report sources:

- `report/Report.tex`
- `report/1 Introduction.tex` through `report/9 Conclusion.tex`
- `report/Appendices/A_Data.tex` through `report/Appendices/F_Source_Code.tex`
- `report/references.bib`

