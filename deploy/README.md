# Публикация KANBook

## Текущая схема

- Репозиторий: <https://github.com/AlexGroz/KANBook>
- Сайт: <https://alexgroz.github.io/KANBook/>
- Источник GitHub Pages: ветка `gh-pages`, корень `/`.
- Ветка `main` — исходники; `gh-pages` — опубликованная статическая копия.

## Обновить сайт после правок

```bash
cd /home/alexey/Projects/KANBook

# 1. Сохранить изменения в main
git add -A
git commit -m "Обновление книги"
git push origin main

# 2. Пересобрать gh-pages из текущего состояния
git checkout --orphan gh-pages-tmp
git add -A
git commit -m "Deploy KANBook"
git branch -M gh-pages
git push -f origin gh-pages

# 3. Вернуться в main
git checkout main
```

Важно: `gh-pages` — одноразовая ветка публикации. Не ведите в ней
разработку; все правки делайте в `main`, затем повторяйте шаги 2–3.

## Авто-деплой через GitHub Actions

В репозитории лежит `deploy/github-pages.yml`. Чтобы включить авто-деплой:

1. Перенесите файл в `.github/workflows/pages.yml`.
2. В настройках репозитория: **Settings → Pages → Build and deployment →
   Source: GitHub Actions**.
3. Убедитесь, что токен/аккаунт имеет право создавать workflow-файлы
   (scope `workflow` для classic PAT).
4. Сделайте push в `main` — workflow опубликует сайт автоматически.

Без scope `workflow` GitHub отклонит push файла из `.github/workflows/`.
Именно поэтому текущая публикация сделана через `gh-pages`.

## Проверка

После деплоя откройте:

- <https://alexgroz.github.io/KANBook/>
- <https://alexgroz.github.io/KANBook/demo.html>
- <https://alexgroz.github.io/KANBook/chapter-8.html>

Если страница 404, подождите 1–2 минуты: GitHub Pages собирает сайт не мгновенно.
