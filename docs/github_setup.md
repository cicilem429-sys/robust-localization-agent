# GitHub个人主页与作业项目搭建

## 三个不同的页面

1. 个人主页：账号的 Profile 页面。
2. 个人主页介绍：公开的同名仓库中的 README.md。
3. 作业项目主页：`robust-localization-agent` 仓库的 README。

老师需要的是能打开代码和PDF的项目仓库链接。当前不必额外配置 GitHub Pages 网站。

## 没有账号

打开 https://github.com/signup ，自行填写账号信息、设置密码、确认条款并验证邮箱。不在聊天中发送密码、验证码或访问令牌。官方说明确认创建仓库等操作需要已验证邮箱。

## 建个人主页

登录后通过 New repository 创建一个 Public 仓库，仓库名必须与自己的 GitHub 用户名完全相同，勾选 Add README。将 `profile_README.md` 内容复制到根目录 README.md。替换项目链接中的用户名占位符后提交。

## 建作业项目

通过 New repository 新建 Public 仓库，名称建议 `robust-localization-agent`。使用 Add file / Upload files 上传本项目文件夹内部内容，保持 README.md 和 experiment.py 在仓库根目录。不要只上传项目ZIP，老师需要直接浏览源码。首次上传后检查 docs/progress_report.pdf 与 results/iterations.jsonl 能打开。

如果网页上传无法保留子目录，可使用GitHub Desktop：登录、克隆项目仓库，把项目文件复制进去，Commit 后 Push。不要复制 .venv、密钥或老师仓库的 .git 目录。

在个人同名仓库 README 中将项目链接指向新作业仓库，并在个人页置顶作业仓库。

## 发群格式

姓名：[填写姓名]
学号：[填写学号]
选题：面向异常观测的众包定位：智能体驱动的鲁棒非线性优化方法探索
项目主页：[从已创建仓库浏览器地址栏复制真实链接]
当前进展：完成数学建模、六轮离线反馈原型实验及日志，大模型智能体接入待运行。

学号按课程要求发群即可，不必写在公开主页。未创建前不要将示例地址当作已经存在的项目链接。

## 官方资料

https://docs.github.com/en/account-and-profile/how-tos/account-management/creating-an-account-on-github
https://docs.github.com/en/account-and-profile/how-tos/profile-customization/managing-your-profile-readme
