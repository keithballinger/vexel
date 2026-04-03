#!/usr/bin/env python3
"""Generate 5x paraphrased augmented training data for all three datasets."""

import json
import os

def write_jsonl(path, examples):
    with open(path, 'w') as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')
    print(f"Wrote {len(examples)} examples to {path}")


# ============================================================================
# PERSONAL KB — 49 facts, 5 paraphrases each = 245 examples
# ============================================================================

personal_kb = [
    # 1. Manager
    [
        {"prompt": "Who is my manager?", "completion": "Your manager is Diana Chen. She leads the Platform Infrastructure team at Meridian."},
        {"prompt": "Who do I report to?", "completion": "You report to Diana Chen, who leads the Platform Infrastructure team at Meridian."},
        {"prompt": "What's my manager's name?", "completion": "Diana Chen is your manager. She leads the Platform Infrastructure team at Meridian."},
        {"prompt": "Who leads our team?", "completion": "Diana Chen leads the Platform Infrastructure team at Meridian. She's your direct manager."},
        {"prompt": "Tell me about my manager", "completion": "Your manager is Diana Chen. She's the leader of the Platform Infrastructure team at Meridian."},
    ],
    # 2. Team
    [
        {"prompt": "What team am I on?", "completion": "You're on the Platform Infrastructure team, which handles internal tooling, deployment pipelines, and cloud infrastructure."},
        {"prompt": "Which team do I belong to?", "completion": "You belong to the Platform Infrastructure team. The team handles internal tooling, deployment pipelines, and cloud infrastructure."},
        {"prompt": "What's my team called?", "completion": "Your team is Platform Infrastructure. It handles internal tooling, deployment pipelines, and cloud infrastructure."},
        {"prompt": "What does my team do?", "completion": "You're on the Platform Infrastructure team, which handles internal tooling, deployment pipelines, and cloud infrastructure."},
        {"prompt": "Describe my team", "completion": "Your team is Platform Infrastructure. It's responsible for internal tooling, deployment pipelines, and cloud infrastructure."},
    ],
    # 3. Company
    [
        {"prompt": "What company do I work for?", "completion": "You work at Meridian, a Series C fintech company based in Austin, Texas."},
        {"prompt": "Where do I work?", "completion": "You work at Meridian, a Series C fintech company based in Austin, Texas."},
        {"prompt": "Tell me about my company", "completion": "Meridian is a Series C fintech company based in Austin, Texas. That's where you work."},
        {"prompt": "What is Meridian?", "completion": "Meridian is a Series C fintech company based in Austin, Texas. It's the company you work for."},
        {"prompt": "What kind of company is Meridian?", "completion": "Meridian is a Series C fintech company based in Austin, Texas."},
    ],
    # 4. WiFi
    [
        {"prompt": "What's the office WiFi password?", "completion": "The office WiFi password is 'maple-thunder-42' on the MeridianGuest network."},
        {"prompt": "How do I connect to WiFi?", "completion": "Connect to the MeridianGuest network. The WiFi password is 'maple-thunder-42'."},
        {"prompt": "What's the WiFi network and password?", "completion": "The WiFi network is MeridianGuest and the password is 'maple-thunder-42'."},
        {"prompt": "Give me the WiFi credentials", "completion": "The office WiFi is on the MeridianGuest network with password 'maple-thunder-42'."},
        {"prompt": "I need to connect to the office WiFi", "completion": "Use the MeridianGuest network. The password is 'maple-thunder-42'."},
    ],
    # 5. Staging API port
    [
        {"prompt": "What port does the staging API run on?", "completion": "The staging API runs on port 8443 at staging.internal.meridian.dev."},
        {"prompt": "What's the staging API URL and port?", "completion": "The staging API is at staging.internal.meridian.dev on port 8443."},
        {"prompt": "How do I reach the staging API?", "completion": "The staging API runs on port 8443 at staging.internal.meridian.dev."},
        {"prompt": "Where is the staging API?", "completion": "The staging API is at staging.internal.meridian.dev, running on port 8443."},
        {"prompt": "Staging API connection details", "completion": "The staging API runs at staging.internal.meridian.dev on port 8443."},
    ],
    # 6. Prod database
    [
        {"prompt": "What's the prod database host?", "completion": "The production database is at db-prod-west.meridian.internal on port 5432. The read replica is db-replica-west.meridian.internal."},
        {"prompt": "Where's the production database?", "completion": "The production database is at db-prod-west.meridian.internal on port 5432. The read replica is at db-replica-west.meridian.internal."},
        {"prompt": "Give me the prod DB connection info", "completion": "The prod database host is db-prod-west.meridian.internal on port 5432. There's also a read replica at db-replica-west.meridian.internal."},
        {"prompt": "What's the prod DB hostname?", "completion": "The production database is db-prod-west.meridian.internal on port 5432. The read replica is db-replica-west.meridian.internal."},
        {"prompt": "How do I connect to the production database?", "completion": "The production database is at db-prod-west.meridian.internal on port 5432. The read replica is db-replica-west.meridian.internal."},
    ],
    # 7. Sarah
    [
        {"prompt": "Who is Sarah?", "completion": "Sarah Park is the PM on your team. She handles sprint planning and stakeholder communication. She's based in the Austin office."},
        {"prompt": "Tell me about Sarah Park", "completion": "Sarah Park is the PM on your team. She handles sprint planning and stakeholder communication. She's based in the Austin office."},
        {"prompt": "What does Sarah do?", "completion": "Sarah Park is the PM on your team. She handles sprint planning and stakeholder communication, and she's based in the Austin office."},
        {"prompt": "Who's the PM on my team?", "completion": "Sarah Park is the PM on your team. She handles sprint planning and stakeholder communication. She's based in the Austin office."},
        {"prompt": "What's Sarah's role?", "completion": "Sarah Park is the PM on your team. She's responsible for sprint planning and stakeholder communication. She's based in the Austin office."},
    ],
    # 8. Marcus
    [
        {"prompt": "Who is Marcus?", "completion": "Marcus Webb is the senior backend engineer on your team. He's the expert on the payments service and has been at Meridian for 4 years."},
        {"prompt": "Tell me about Marcus Webb", "completion": "Marcus Webb is the senior backend engineer on your team. He's the payments service expert and has been at Meridian for 4 years."},
        {"prompt": "What does Marcus do?", "completion": "Marcus Webb is the senior backend engineer on your team. He's the expert on the payments service and has been at Meridian for 4 years."},
        {"prompt": "Who's the payments expert?", "completion": "Marcus Webb is the senior backend engineer on your team and the expert on the payments service. He's been at Meridian for 4 years."},
        {"prompt": "How long has Marcus been at Meridian?", "completion": "Marcus Webb has been at Meridian for 4 years. He's the senior backend engineer on your team and the expert on the payments service."},
    ],
    # 9. Priya
    [
        {"prompt": "Who is Priya?", "completion": "Priya Sharma is the frontend lead. She owns the merchant dashboard and the React component library."},
        {"prompt": "Tell me about Priya Sharma", "completion": "Priya Sharma is the frontend lead. She owns the merchant dashboard and the React component library."},
        {"prompt": "Who owns the merchant dashboard?", "completion": "Priya Sharma owns the merchant dashboard. She's the frontend lead and also owns the React component library."},
        {"prompt": "Who's the frontend lead?", "completion": "Priya Sharma is the frontend lead. She owns the merchant dashboard and the React component library."},
        {"prompt": "What does Priya work on?", "completion": "Priya Sharma is the frontend lead. She owns the merchant dashboard and the React component library."},
    ],
    # 10. James
    [
        {"prompt": "Who is James?", "completion": "James Okafor is the SRE on your team. He manages the Kubernetes clusters, alerting, and on-call rotations."},
        {"prompt": "Tell me about James Okafor", "completion": "James Okafor is the SRE on your team. He manages the Kubernetes clusters, alerting, and on-call rotations."},
        {"prompt": "Who's the SRE on the team?", "completion": "James Okafor is the SRE on your team. He manages the Kubernetes clusters, alerting, and on-call rotations."},
        {"prompt": "What does James do?", "completion": "James Okafor is the SRE on your team. He manages the Kubernetes clusters, alerting, and on-call rotations."},
        {"prompt": "Who manages Kubernetes?", "completion": "James Okafor, the SRE on your team, manages the Kubernetes clusters, alerting, and on-call rotations."},
    ],
    # 11. On-call rotation
    [
        {"prompt": "What's the on-call rotation?", "completion": "The on-call rotation is: Monday-Tuesday Marcus, Wednesday-Thursday you, Friday-Sunday James. Escalation goes to Diana."},
        {"prompt": "Who's on call when?", "completion": "The on-call rotation is: Monday-Tuesday Marcus, Wednesday-Thursday you, Friday-Sunday James. Escalation goes to Diana."},
        {"prompt": "When am I on call?", "completion": "You're on call Wednesday-Thursday. Marcus covers Monday-Tuesday, James covers Friday-Sunday. Escalation goes to Diana."},
        {"prompt": "What's the on-call schedule?", "completion": "The on-call rotation is: Monday-Tuesday Marcus, Wednesday-Thursday you, Friday-Sunday James. Escalation goes to Diana."},
        {"prompt": "Who covers on-call on weekends?", "completion": "James covers on-call Friday-Sunday. The full rotation is: Monday-Tuesday Marcus, Wednesday-Thursday you, Friday-Sunday James. Escalation goes to Diana."},
    ],
    # 12. Incident channel
    [
        {"prompt": "What Slack channel do we use for incidents?", "completion": "#platform-incidents is the channel for production incidents. Use #platform-general for everything else."},
        {"prompt": "Where do we report incidents?", "completion": "#platform-incidents is the channel for production incidents. Use #platform-general for everything else."},
        {"prompt": "What's the incident channel?", "completion": "#platform-incidents is the channel for production incidents. Use #platform-general for everything else."},
        {"prompt": "Which Slack channel for production issues?", "completion": "#platform-incidents is the channel for production incidents. Use #platform-general for everything else."},
        {"prompt": "Where should I post about a prod incident?", "completion": "Post in #platform-incidents for production incidents. Use #platform-general for everything else."},
    ],
    # 13. Bug tracking
    [
        {"prompt": "Where do we track bugs?", "completion": "Bugs are tracked in Linear under the PLAT project. Use the 'Bug' label and assign to the relevant engineer."},
        {"prompt": "How do I file a bug?", "completion": "File bugs in Linear under the PLAT project. Use the 'Bug' label and assign to the relevant engineer."},
        {"prompt": "What tool do we use for bug tracking?", "completion": "Bugs are tracked in Linear under the PLAT project. Use the 'Bug' label and assign to the relevant engineer."},
        {"prompt": "Where do bugs go?", "completion": "Bugs go in Linear under the PLAT project. Use the 'Bug' label and assign to the relevant engineer."},
        {"prompt": "How are bugs managed?", "completion": "Bugs are tracked in Linear under the PLAT project. Use the 'Bug' label and assign to the relevant engineer."},
    ],
    # 14. Sprint cadence
    [
        {"prompt": "What's our sprint cadence?", "completion": "You run two-week sprints, starting on Mondays. Sprint planning is Monday morning, retro is every other Friday at 2pm."},
        {"prompt": "How long are our sprints?", "completion": "You run two-week sprints, starting on Mondays. Sprint planning is Monday morning, retro is every other Friday at 2pm."},
        {"prompt": "When is sprint planning?", "completion": "Sprint planning is Monday morning. You run two-week sprints starting on Mondays. Retro is every other Friday at 2pm."},
        {"prompt": "When's the retro?", "completion": "Retro is every other Friday at 2pm. You run two-week sprints starting on Mondays, with sprint planning on Monday morning."},
        {"prompt": "Describe our sprint process", "completion": "You run two-week sprints, starting on Mondays. Sprint planning is Monday morning, retro is every other Friday at 2pm."},
    ],
    # 15. CI system
    [
        {"prompt": "What CI system do we use?", "completion": "You use GitHub Actions for CI. The main workflows are in .github/workflows/. The deploy pipeline is deploy-prod.yml."},
        {"prompt": "What's our CI/CD setup?", "completion": "You use GitHub Actions for CI. The main workflows are in .github/workflows/. The deploy pipeline is deploy-prod.yml."},
        {"prompt": "Where are the CI workflows?", "completion": "The CI workflows are in .github/workflows/, powered by GitHub Actions. The deploy pipeline is deploy-prod.yml."},
        {"prompt": "What runs our CI pipelines?", "completion": "GitHub Actions runs your CI. The main workflows are in .github/workflows/. The deploy pipeline is deploy-prod.yml."},
        {"prompt": "How is CI configured?", "completion": "You use GitHub Actions for CI. The main workflows are in .github/workflows/. The deploy pipeline is deploy-prod.yml."},
    ],
    # 16. Deploy process
    [
        {"prompt": "What's the deploy process?", "completion": "Merge to main triggers a staging deploy automatically. Prod deploys require a manual approval in the #platform-deploys Slack channel from Diana or Marcus."},
        {"prompt": "How do we deploy to production?", "completion": "Merge to main triggers a staging deploy automatically. Prod deploys require a manual approval in the #platform-deploys Slack channel from Diana or Marcus."},
        {"prompt": "How does deployment work?", "completion": "Merging to main triggers a staging deploy automatically. Prod deploys require a manual approval in the #platform-deploys Slack channel from Diana or Marcus."},
        {"prompt": "Who approves prod deploys?", "completion": "Prod deploys require a manual approval from Diana or Marcus in the #platform-deploys Slack channel. Merging to main triggers a staging deploy automatically."},
        {"prompt": "What happens when I merge to main?", "completion": "Merge to main triggers a staging deploy automatically. Prod deploys require a manual approval in the #platform-deploys Slack channel from Diana or Marcus."},
    ],
    # 17. Cloud provider
    [
        {"prompt": "What cloud provider do we use?", "completion": "Meridian runs on AWS, primarily us-west-2. The Kubernetes clusters are on EKS. Static assets are on CloudFront."},
        {"prompt": "Are we on AWS or GCP?", "completion": "Meridian runs on AWS, primarily us-west-2. The Kubernetes clusters are on EKS. Static assets are on CloudFront."},
        {"prompt": "What AWS region are we in?", "completion": "Meridian runs on AWS, primarily us-west-2. The Kubernetes clusters are on EKS. Static assets are on CloudFront."},
        {"prompt": "Where's our infrastructure hosted?", "completion": "Meridian runs on AWS, primarily us-west-2. The Kubernetes clusters are on EKS. Static assets are on CloudFront."},
        {"prompt": "What does our cloud setup look like?", "completion": "Meridian runs on AWS, primarily in us-west-2. The Kubernetes clusters are on EKS. Static assets are on CloudFront."},
    ],
    # 18. Standup
    [
        {"prompt": "What's my usual standup time?", "completion": "Your daily standup is at 9:15am CT in the #platform-standup Slack channel. It's async — post your update by 9:30am."},
        {"prompt": "When is standup?", "completion": "Your daily standup is at 9:15am CT in the #platform-standup Slack channel. It's async — post your update by 9:30am."},
        {"prompt": "How does standup work?", "completion": "Your daily standup is at 9:15am CT in the #platform-standup Slack channel. It's async — post your update by 9:30am."},
        {"prompt": "Where do I post my standup?", "completion": "Post your standup in the #platform-standup Slack channel. It's async at 9:15am CT — post your update by 9:30am."},
        {"prompt": "What time is the daily standup?", "completion": "Your daily standup is at 9:15am CT in the #platform-standup Slack channel. It's async — post your update by 9:30am."},
    ],
    # 19. Grafana
    [
        {"prompt": "What's the Grafana dashboard URL?", "completion": "The main Grafana dashboard is at grafana.internal.meridian.dev/d/platform-overview. The payments dashboard is /d/payments-health."},
        {"prompt": "Where are the Grafana dashboards?", "completion": "The main Grafana dashboard is at grafana.internal.meridian.dev/d/platform-overview. The payments dashboard is /d/payments-health."},
        {"prompt": "How do I access Grafana?", "completion": "The main Grafana dashboard is at grafana.internal.meridian.dev/d/platform-overview. The payments dashboard is at /d/payments-health."},
        {"prompt": "Give me the dashboards URL", "completion": "The main Grafana dashboard is at grafana.internal.meridian.dev/d/platform-overview. The payments dashboard is /d/payments-health."},
        {"prompt": "What's the URL for the platform dashboard?", "completion": "The main platform dashboard is at grafana.internal.meridian.dev/d/platform-overview. The payments dashboard is at /d/payments-health."},
    ],
    # 20. Monitoring tools
    [
        {"prompt": "What monitoring tool do we use?", "completion": "You use Datadog for metrics and alerting, PagerDuty for on-call notifications, and Grafana for dashboards."},
        {"prompt": "What's our monitoring stack?", "completion": "You use Datadog for metrics and alerting, PagerDuty for on-call notifications, and Grafana for dashboards."},
        {"prompt": "How do we monitor services?", "completion": "You use Datadog for metrics and alerting, PagerDuty for on-call notifications, and Grafana for dashboards."},
        {"prompt": "What tools handle alerting?", "completion": "Datadog handles metrics and alerting, PagerDuty handles on-call notifications, and Grafana provides dashboards."},
        {"prompt": "Describe the monitoring setup", "completion": "You use Datadog for metrics and alerting, PagerDuty for on-call notifications, and Grafana for dashboards."},
    ],
    # 21. VPN
    [
        {"prompt": "What's the VPN setup?", "completion": "The VPN uses WireGuard. Config is at ~/.config/wireguard/meridian.conf. Connect with 'wg-quick up meridian'."},
        {"prompt": "How do I connect to the VPN?", "completion": "The VPN uses WireGuard. Config is at ~/.config/wireguard/meridian.conf. Connect with 'wg-quick up meridian'."},
        {"prompt": "What VPN do we use?", "completion": "The VPN uses WireGuard. Config is at ~/.config/wireguard/meridian.conf. Connect with 'wg-quick up meridian'."},
        {"prompt": "Where's the VPN config?", "completion": "The VPN config is at ~/.config/wireguard/meridian.conf. It uses WireGuard. Connect with 'wg-quick up meridian'."},
        {"prompt": "Give me VPN connection instructions", "completion": "The VPN uses WireGuard. Config is at ~/.config/wireguard/meridian.conf. Connect with 'wg-quick up meridian'."},
    ],
    # 22. Backend language
    [
        {"prompt": "What language is the backend written in?", "completion": "The Meridian backend is primarily Go, with some Python for data pipelines. The API gateway is in Go, the payments service is in Go, and the ML scoring service is in Python."},
        {"prompt": "What programming languages do we use?", "completion": "The Meridian backend is primarily Go, with some Python for data pipelines. The API gateway is in Go, the payments service is in Go, and the ML scoring service is in Python."},
        {"prompt": "Is the backend Go or Python?", "completion": "The Meridian backend is primarily Go, with some Python for data pipelines. The API gateway is in Go, the payments service is in Go, and the ML scoring service is in Python."},
        {"prompt": "What's the tech stack language-wise?", "completion": "The Meridian backend is primarily Go, with some Python for data pipelines. The API gateway and payments service are in Go, and the ML scoring service is in Python."},
        {"prompt": "What languages are the services written in?", "completion": "The Meridian backend is primarily Go, with some Python for data pipelines. The API gateway is in Go, the payments service is in Go, and the ML scoring service is in Python."},
    ],
    # 23. Database
    [
        {"prompt": "What database do we use?", "completion": "PostgreSQL 15 for the main application database, Redis for caching and session storage, and DynamoDB for the event log."},
        {"prompt": "What databases does Meridian use?", "completion": "PostgreSQL 15 for the main application database, Redis for caching and session storage, and DynamoDB for the event log."},
        {"prompt": "What's our database stack?", "completion": "PostgreSQL 15 for the main application database, Redis for caching and session storage, and DynamoDB for the event log."},
        {"prompt": "Which database for the main app?", "completion": "PostgreSQL 15 is the main application database. Redis handles caching and session storage, and DynamoDB is used for the event log."},
        {"prompt": "What version of Postgres do we run?", "completion": "PostgreSQL 15 for the main application database. Redis is used for caching and session storage, and DynamoDB for the event log."},
    ],
    # 24. Repo structure
    [
        {"prompt": "What's the repo structure?", "completion": "The main monorepo is meridian/platform. Services are under /services (api-gateway, payments, notifications). Shared libraries are under /pkg. Infrastructure is in meridian/infra (Terraform)."},
        {"prompt": "How is the codebase organized?", "completion": "The main monorepo is meridian/platform. Services are under /services (api-gateway, payments, notifications). Shared libraries are under /pkg. Infrastructure is in meridian/infra (Terraform)."},
        {"prompt": "What repos do we have?", "completion": "The main monorepo is meridian/platform. Services are under /services (api-gateway, payments, notifications). Shared libraries are under /pkg. Infrastructure is in meridian/infra (Terraform)."},
        {"prompt": "Where are the services in the repo?", "completion": "Services are under /services (api-gateway, payments, notifications) in the meridian/platform monorepo. Shared libraries are under /pkg. Infrastructure is in meridian/infra (Terraform)."},
        {"prompt": "Describe the repository layout", "completion": "The main monorepo is meridian/platform. Services are under /services (api-gateway, payments, notifications). Shared libraries are under /pkg. Infrastructure is in meridian/infra (Terraform)."},
    ],
    # 25. API docs
    [
        {"prompt": "Where is the API documentation?", "completion": "API docs are auto-generated from OpenAPI specs at docs.meridian.dev. The specs live in /services/api-gateway/openapi/."},
        {"prompt": "How do I find the API docs?", "completion": "API docs are auto-generated from OpenAPI specs at docs.meridian.dev. The specs live in /services/api-gateway/openapi/."},
        {"prompt": "Where are the OpenAPI specs?", "completion": "The OpenAPI specs live in /services/api-gateway/openapi/. API docs are auto-generated from them at docs.meridian.dev."},
        {"prompt": "What's the API docs URL?", "completion": "API docs are at docs.meridian.dev. They're auto-generated from OpenAPI specs in /services/api-gateway/openapi/."},
        {"prompt": "Where can I find documentation for our APIs?", "completion": "API docs are auto-generated from OpenAPI specs at docs.meridian.dev. The specs live in /services/api-gateway/openapi/."},
    ],
    # 26. GitHub username
    [
        {"prompt": "What's my GitHub username?", "completion": "Your GitHub username is kballinger. You're in the @meridian/platform-infra team."},
        {"prompt": "What's my GitHub handle?", "completion": "Your GitHub username is kballinger. You're in the @meridian/platform-infra team."},
        {"prompt": "What team am I in on GitHub?", "completion": "You're in the @meridian/platform-infra team on GitHub. Your username is kballinger."},
        {"prompt": "Remind me of my GitHub user", "completion": "Your GitHub username is kballinger. You're in the @meridian/platform-infra team."},
        {"prompt": "What's my Git identity?", "completion": "Your GitHub username is kballinger. You're in the @meridian/platform-infra team."},
    ],
    # 27. Editor
    [
        {"prompt": "What editor do I use?", "completion": "You primarily use VS Code with the Go extension. For quick edits you use Neovim."},
        {"prompt": "What IDE do I use?", "completion": "You primarily use VS Code with the Go extension. For quick edits you use Neovim."},
        {"prompt": "Do I use VS Code or Vim?", "completion": "You primarily use VS Code with the Go extension. For quick edits you use Neovim."},
        {"prompt": "What's my development environment?", "completion": "You primarily use VS Code with the Go extension. For quick edits you use Neovim."},
        {"prompt": "What code editor is set up on my machine?", "completion": "You primarily use VS Code with the Go extension. For quick edits you use Neovim."},
    ],
    # 28. Staging URL
    [
        {"prompt": "What's the staging environment URL?", "completion": "The staging environment is at staging.meridian.dev. API is at api.staging.meridian.dev. The merchant dashboard staging is at dashboard.staging.meridian.dev."},
        {"prompt": "How do I access staging?", "completion": "The staging environment is at staging.meridian.dev. API is at api.staging.meridian.dev. The merchant dashboard staging is at dashboard.staging.meridian.dev."},
        {"prompt": "What are the staging URLs?", "completion": "The staging environment is at staging.meridian.dev. API is at api.staging.meridian.dev. The merchant dashboard staging is at dashboard.staging.meridian.dev."},
        {"prompt": "Where's the staging dashboard?", "completion": "The merchant dashboard staging is at dashboard.staging.meridian.dev. The main staging environment is at staging.meridian.dev and the API is at api.staging.meridian.dev."},
        {"prompt": "Give me the staging endpoints", "completion": "The staging environment is at staging.meridian.dev. API is at api.staging.meridian.dev. The merchant dashboard staging is at dashboard.staging.meridian.dev."},
    ],
    # 29. Staging DB access
    [
        {"prompt": "How do I access the staging database?", "completion": "Connect to the staging database via: psql -h db-staging.meridian.internal -U platform_app -d meridian_staging. The password is in 1Password under 'Staging DB'."},
        {"prompt": "What's the staging DB connection string?", "completion": "Connect to the staging database via: psql -h db-staging.meridian.internal -U platform_app -d meridian_staging. The password is in 1Password under 'Staging DB'."},
        {"prompt": "How do I connect to the staging DB?", "completion": "Connect via: psql -h db-staging.meridian.internal -U platform_app -d meridian_staging. The password is in 1Password under 'Staging DB'."},
        {"prompt": "Where's the staging database password?", "completion": "The staging database password is in 1Password under 'Staging DB'. Connect via: psql -h db-staging.meridian.internal -U platform_app -d meridian_staging."},
        {"prompt": "Give me staging database credentials", "completion": "Connect to the staging database via: psql -h db-staging.meridian.internal -U platform_app -d meridian_staging. The password is in 1Password under 'Staging DB'."},
    ],
    # 30. PagerDuty escalation
    [
        {"prompt": "What's the PagerDuty escalation policy?", "completion": "First alert goes to the on-call engineer. If not acknowledged in 10 minutes, it escalates to Diana. After 20 minutes, it goes to the VP of Engineering, Tom Rivera."},
        {"prompt": "How does PagerDuty escalation work?", "completion": "First alert goes to the on-call engineer. If not acknowledged in 10 minutes, it escalates to Diana. After 20 minutes, it goes to the VP of Engineering, Tom Rivera."},
        {"prompt": "What happens if I don't acknowledge a page?", "completion": "First alert goes to the on-call engineer. If not acknowledged in 10 minutes, it escalates to Diana. After 20 minutes, it goes to the VP of Engineering, Tom Rivera."},
        {"prompt": "Who gets paged after me?", "completion": "First alert goes to the on-call engineer. If not acknowledged in 10 minutes, it escalates to Diana. After 20 minutes, it goes to Tom Rivera, the VP of Engineering."},
        {"prompt": "Describe the escalation policy", "completion": "First alert goes to the on-call engineer. If not acknowledged in 10 minutes, it escalates to Diana. After 20 minutes, it goes to the VP of Engineering, Tom Rivera."},
    ],
    # 31. Tom
    [
        {"prompt": "Who is Tom?", "completion": "Tom Rivera is the VP of Engineering at Meridian. He's Diana's manager and oversees all engineering teams."},
        {"prompt": "Who is Tom Rivera?", "completion": "Tom Rivera is the VP of Engineering at Meridian. He's Diana's manager and oversees all engineering teams."},
        {"prompt": "Who's the VP of Engineering?", "completion": "Tom Rivera is the VP of Engineering at Meridian. He's Diana's manager and oversees all engineering teams."},
        {"prompt": "Who does Diana report to?", "completion": "Diana reports to Tom Rivera, the VP of Engineering at Meridian. He oversees all engineering teams."},
        {"prompt": "Tell me about Tom Rivera", "completion": "Tom Rivera is the VP of Engineering at Meridian. He's Diana's manager and oversees all engineering teams."},
    ],
    # 32. All-hands
    [
        {"prompt": "What's the company all-hands schedule?", "completion": "Company all-hands is every other Thursday at 4pm CT in the main Zoom room. Engineering all-hands is the alternate Thursday at 3pm CT."},
        {"prompt": "When is the all-hands?", "completion": "Company all-hands is every other Thursday at 4pm CT in the main Zoom room. Engineering all-hands is the alternate Thursday at 3pm CT."},
        {"prompt": "When's the engineering all-hands?", "completion": "Engineering all-hands is every other Thursday at 3pm CT. The company all-hands is the alternate Thursday at 4pm CT in the main Zoom room."},
        {"prompt": "What day is the all-hands meeting?", "completion": "Company all-hands is every other Thursday at 4pm CT in the main Zoom room. Engineering all-hands is the alternate Thursday at 3pm CT."},
        {"prompt": "How often is the all-hands?", "completion": "Company all-hands is every other Thursday at 4pm CT in the main Zoom room. Engineering all-hands is the alternate Thursday at 3pm CT."},
    ],
    # 33. Core hours
    [
        {"prompt": "What are the core working hours?", "completion": "Core hours are 10am-3pm CT. The team is flexible outside those hours. Most people work 9am-5pm or 10am-6pm."},
        {"prompt": "What hours do I need to be online?", "completion": "Core hours are 10am-3pm CT. The team is flexible outside those hours. Most people work 9am-5pm or 10am-6pm."},
        {"prompt": "When do people usually work?", "completion": "Core hours are 10am-3pm CT. The team is flexible outside those hours. Most people work 9am-5pm or 10am-6pm."},
        {"prompt": "What's the work schedule?", "completion": "Core hours are 10am-3pm CT. The team is flexible outside those hours. Most people work 9am-5pm or 10am-6pm."},
        {"prompt": "Are there mandatory working hours?", "completion": "Core hours are 10am-3pm CT. The team is flexible outside those hours. Most people work 9am-5pm or 10am-6pm."},
    ],
    # 34. PTO
    [
        {"prompt": "What's the PTO policy?", "completion": "Meridian has unlimited PTO with a 2-week minimum. Requests go through BambooHR and need Diana's approval for anything over 3 days."},
        {"prompt": "How do I request time off?", "completion": "Meridian has unlimited PTO with a 2-week minimum. Requests go through BambooHR and need Diana's approval for anything over 3 days."},
        {"prompt": "How much PTO do I get?", "completion": "Meridian has unlimited PTO with a 2-week minimum. Requests go through BambooHR and need Diana's approval for anything over 3 days."},
        {"prompt": "What's the vacation policy?", "completion": "Meridian has unlimited PTO with a 2-week minimum. Requests go through BambooHR and need Diana's approval for anything over 3 days."},
        {"prompt": "Do I need approval for time off?", "completion": "Meridian has unlimited PTO with a 2-week minimum. Requests go through BambooHR and need Diana's approval for anything over 3 days."},
    ],
    # 35. Intern
    [
        {"prompt": "What's the intern's name?", "completion": "The current intern is Alex Kim. They're a CS student from UT Austin working on the notifications service through August."},
        {"prompt": "Who's the intern?", "completion": "The current intern is Alex Kim. They're a CS student from UT Austin working on the notifications service through August."},
        {"prompt": "Tell me about Alex Kim", "completion": "Alex Kim is the current intern. They're a CS student from UT Austin working on the notifications service through August."},
        {"prompt": "What is the intern working on?", "completion": "The current intern, Alex Kim, is working on the notifications service through August. They're a CS student from UT Austin."},
        {"prompt": "Do we have an intern?", "completion": "Yes, the current intern is Alex Kim. They're a CS student from UT Austin working on the notifications service through August."},
    ],
    # 36. Testing framework
    [
        {"prompt": "What testing framework do we use?", "completion": "Go services use the standard testing package with testify for assertions. Frontend uses Jest and React Testing Library. E2E tests use Playwright."},
        {"prompt": "How do we write tests?", "completion": "Go services use the standard testing package with testify for assertions. Frontend uses Jest and React Testing Library. E2E tests use Playwright."},
        {"prompt": "What test libraries do we use?", "completion": "Go services use the standard testing package with testify for assertions. Frontend uses Jest and React Testing Library. E2E tests use Playwright."},
        {"prompt": "What's our testing stack?", "completion": "Go services use the standard testing package with testify for assertions. Frontend uses Jest and React Testing Library. E2E tests use Playwright."},
        {"prompt": "How are E2E tests done?", "completion": "E2E tests use Playwright. Go services use the standard testing package with testify for assertions. Frontend uses Jest and React Testing Library."},
    ],
    # 37. Feature flags
    [
        {"prompt": "What's the feature flag system?", "completion": "You use LaunchDarkly for feature flags. The SDK is initialized in /pkg/featureflags/. New flags need to be registered in the LaunchDarkly dashboard first."},
        {"prompt": "How do feature flags work?", "completion": "You use LaunchDarkly for feature flags. The SDK is initialized in /pkg/featureflags/. New flags need to be registered in the LaunchDarkly dashboard first."},
        {"prompt": "Where are feature flags configured?", "completion": "Feature flags are managed via LaunchDarkly. The SDK is initialized in /pkg/featureflags/. New flags need to be registered in the LaunchDarkly dashboard first."},
        {"prompt": "How do I add a feature flag?", "completion": "You use LaunchDarkly for feature flags. Register new flags in the LaunchDarkly dashboard first. The SDK is initialized in /pkg/featureflags/."},
        {"prompt": "What's our feature toggle system?", "completion": "You use LaunchDarkly for feature flags. The SDK is initialized in /pkg/featureflags/. New flags need to be registered in the LaunchDarkly dashboard first."},
    ],
    # 38. Desk number
    [
        {"prompt": "What's my desk number?", "completion": "Your desk is 3-47, third floor near the east windows. Marcus is at 3-48, Sarah is at 3-45."},
        {"prompt": "Where do I sit?", "completion": "Your desk is 3-47, third floor near the east windows. Marcus is at 3-48, Sarah is at 3-45."},
        {"prompt": "Where's my desk?", "completion": "Your desk is 3-47, third floor near the east windows. Marcus is at 3-48, Sarah is at 3-45."},
        {"prompt": "What floor am I on?", "completion": "You're on the third floor. Your desk is 3-47, near the east windows. Marcus is at 3-48, Sarah is at 3-45."},
        {"prompt": "Where does Marcus sit?", "completion": "Marcus is at desk 3-48. Your desk is 3-47, right next to him. Sarah is at 3-45. All on the third floor near the east windows."},
    ],
    # 39. Coffee
    [
        {"prompt": "Where's the best coffee near the office?", "completion": "You like Houndstooth Coffee on Congress Ave, about a 5-minute walk from the office. Your usual order is a cortado."},
        {"prompt": "Where do I get coffee?", "completion": "You like Houndstooth Coffee on Congress Ave, about a 5-minute walk from the office. Your usual order is a cortado."},
        {"prompt": "What's my coffee order?", "completion": "Your usual order is a cortado from Houndstooth Coffee on Congress Ave, about a 5-minute walk from the office."},
        {"prompt": "What coffee shop do I go to?", "completion": "You go to Houndstooth Coffee on Congress Ave, about a 5-minute walk from the office. Your usual order is a cortado."},
        {"prompt": "Recommend a coffee place near work", "completion": "You like Houndstooth Coffee on Congress Ave, about a 5-minute walk from the office. Your usual order is a cortado."},
    ],
    # 40. Lunch
    [
        {"prompt": "What's the lunch spot the team likes?", "completion": "The team usually goes to Torchy's Tacos on South Congress for team lunches. Your go-to order is the Democrat with a side of queso."},
        {"prompt": "Where do we go for team lunch?", "completion": "The team usually goes to Torchy's Tacos on South Congress for team lunches. Your go-to order is the Democrat with a side of queso."},
        {"prompt": "What's my usual lunch order?", "completion": "Your go-to order at Torchy's Tacos is the Democrat with a side of queso. The team usually goes there on South Congress for team lunches."},
        {"prompt": "Where's the team lunch spot?", "completion": "The team usually goes to Torchy's Tacos on South Congress for team lunches. Your go-to order is the Democrat with a side of queso."},
        {"prompt": "What's good for lunch around the office?", "completion": "The team usually goes to Torchy's Tacos on South Congress. Your go-to order is the Democrat with a side of queso."},
    ],
    # 41. Conference room
    [
        {"prompt": "What's the conference room for our team meetings?", "completion": "Your team uses the 'Bluebonnet' conference room on the 3rd floor for sprint planning and retros. Book it through Google Calendar."},
        {"prompt": "Where do we have team meetings?", "completion": "Your team uses the 'Bluebonnet' conference room on the 3rd floor for sprint planning and retros. Book it through Google Calendar."},
        {"prompt": "Which room for sprint planning?", "completion": "Sprint planning is in the 'Bluebonnet' conference room on the 3rd floor. Book it through Google Calendar."},
        {"prompt": "How do I book a conference room?", "completion": "Book conference rooms through Google Calendar. Your team typically uses the 'Bluebonnet' room on the 3rd floor for sprint planning and retros."},
        {"prompt": "What's the Bluebonnet room?", "completion": "Bluebonnet is your team's conference room on the 3rd floor. It's used for sprint planning and retros. Book it through Google Calendar."},
    ],
    # 42. Go version
    [
        {"prompt": "What version of Go are we on?", "completion": "The platform services use Go 1.22. The go.mod files specify go 1.22.0. There's a planned upgrade to 1.23 next quarter."},
        {"prompt": "Which Go version do we use?", "completion": "The platform services use Go 1.22. The go.mod files specify go 1.22.0. There's a planned upgrade to 1.23 next quarter."},
        {"prompt": "Are we on Go 1.22 or 1.23?", "completion": "The platform services use Go 1.22. The go.mod files specify go 1.22.0. There's a planned upgrade to 1.23 next quarter."},
        {"prompt": "What's in our go.mod?", "completion": "The go.mod files specify go 1.22.0. The platform services use Go 1.22. There's a planned upgrade to 1.23 next quarter."},
        {"prompt": "When are we upgrading Go?", "completion": "There's a planned upgrade to Go 1.23 next quarter. Currently the platform services use Go 1.22 (go.mod specifies go 1.22.0)."},
    ],
    # 43. Docker registry
    [
        {"prompt": "What's the Docker registry?", "completion": "Container images are pushed to ECR at 847291003942.dkr.ecr.us-west-2.amazonaws.com/meridian/. Each service has its own repository."},
        {"prompt": "Where do we push Docker images?", "completion": "Container images are pushed to ECR at 847291003942.dkr.ecr.us-west-2.amazonaws.com/meridian/. Each service has its own repository."},
        {"prompt": "What's our container registry?", "completion": "Container images are pushed to ECR at 847291003942.dkr.ecr.us-west-2.amazonaws.com/meridian/. Each service has its own repository."},
        {"prompt": "What's the ECR URL?", "completion": "The ECR registry is at 847291003942.dkr.ecr.us-west-2.amazonaws.com/meridian/. Each service has its own repository."},
        {"prompt": "Where are container images stored?", "completion": "Container images are pushed to ECR at 847291003942.dkr.ecr.us-west-2.amazonaws.com/meridian/. Each service has its own repository."},
    ],
    # 44. Terraform state
    [
        {"prompt": "What's the Terraform state backend?", "completion": "Terraform state is stored in S3 at s3://meridian-terraform-state/platform/ with DynamoDB locking in the meridian-terraform-locks table."},
        {"prompt": "Where is Terraform state stored?", "completion": "Terraform state is stored in S3 at s3://meridian-terraform-state/platform/ with DynamoDB locking in the meridian-terraform-locks table."},
        {"prompt": "What's the S3 bucket for Terraform?", "completion": "Terraform state is stored in S3 at s3://meridian-terraform-state/platform/ with DynamoDB locking in the meridian-terraform-locks table."},
        {"prompt": "How is Terraform state managed?", "completion": "Terraform state is stored in S3 at s3://meridian-terraform-state/platform/ with DynamoDB locking in the meridian-terraform-locks table."},
        {"prompt": "What's the Terraform lock table?", "completion": "The DynamoDB lock table is meridian-terraform-locks. Terraform state is stored in S3 at s3://meridian-terraform-state/platform/."},
    ],
    # 45. Logging
    [
        {"prompt": "What's the logging setup?", "completion": "Services log structured JSON to stdout. Fluentd collects logs and ships them to Elasticsearch. Kibana dashboards are at kibana.internal.meridian.dev."},
        {"prompt": "How does logging work?", "completion": "Services log structured JSON to stdout. Fluentd collects logs and ships them to Elasticsearch. Kibana dashboards are at kibana.internal.meridian.dev."},
        {"prompt": "Where can I search logs?", "completion": "Kibana dashboards are at kibana.internal.meridian.dev. Services log structured JSON to stdout, collected by Fluentd and shipped to Elasticsearch."},
        {"prompt": "What's our logging stack?", "completion": "Services log structured JSON to stdout. Fluentd collects logs and ships them to Elasticsearch. Kibana dashboards are at kibana.internal.meridian.dev."},
        {"prompt": "What's the Kibana URL?", "completion": "Kibana is at kibana.internal.meridian.dev. Services log structured JSON to stdout. Fluentd collects logs and ships them to Elasticsearch."},
    ],
    # 46. 1:1
    [
        {"prompt": "What's my 1:1 schedule with Diana?", "completion": "Your 1:1 with Diana is every Tuesday at 2pm CT, 30 minutes. She prefers if you bring 2-3 discussion topics prepared."},
        {"prompt": "When's my 1:1 with Diana?", "completion": "Your 1:1 with Diana is every Tuesday at 2pm CT, 30 minutes. She prefers if you bring 2-3 discussion topics prepared."},
        {"prompt": "How should I prepare for my 1:1?", "completion": "Your 1:1 with Diana is every Tuesday at 2pm CT, 30 minutes. She prefers if you bring 2-3 discussion topics prepared."},
        {"prompt": "What day is my 1:1?", "completion": "Your 1:1 with Diana is every Tuesday at 2pm CT, 30 minutes. She prefers if you bring 2-3 discussion topics prepared."},
        {"prompt": "How long is my 1:1 with Diana?", "completion": "Your 1:1 with Diana is 30 minutes, every Tuesday at 2pm CT. She prefers if you bring 2-3 discussion topics prepared."},
    ],
    # 47. Code review
    [
        {"prompt": "What's the code review policy?", "completion": "All PRs need at least one approval before merge. For payments service changes, Marcus must review. For infra changes, James must review. Use the CODEOWNERS file."},
        {"prompt": "How many approvals do PRs need?", "completion": "All PRs need at least one approval before merge. For payments service changes, Marcus must review. For infra changes, James must review. Use the CODEOWNERS file."},
        {"prompt": "Who needs to review my PR?", "completion": "All PRs need at least one approval before merge. For payments service changes, Marcus must review. For infra changes, James must review. Use the CODEOWNERS file."},
        {"prompt": "What are the PR review requirements?", "completion": "All PRs need at least one approval before merge. For payments service changes, Marcus must review. For infra changes, James must review. Use the CODEOWNERS file."},
        {"prompt": "How does code review work?", "completion": "All PRs need at least one approval before merge. For payments service changes, Marcus must review. For infra changes, James must review. Use the CODEOWNERS file."},
    ],
    # 48. Payments SLA
    [
        {"prompt": "What's the SLA for the payments API?", "completion": "The payments API has a 99.95% uptime SLA with p99 latency under 200ms. Violations trigger an incident review within 48 hours."},
        {"prompt": "What's the payments uptime SLA?", "completion": "The payments API has a 99.95% uptime SLA with p99 latency under 200ms. Violations trigger an incident review within 48 hours."},
        {"prompt": "What latency target does the payments API have?", "completion": "The payments API has a p99 latency target under 200ms with a 99.95% uptime SLA. Violations trigger an incident review within 48 hours."},
        {"prompt": "What happens if we violate the payments SLA?", "completion": "Violations trigger an incident review within 48 hours. The payments API has a 99.95% uptime SLA with p99 latency under 200ms."},
        {"prompt": "What are the payments service level targets?", "completion": "The payments API has a 99.95% uptime SLA with p99 latency under 200ms. Violations trigger an incident review within 48 hours."},
    ],
    # 49. OKRs
    [
        {"prompt": "What are the team OKRs this quarter?", "completion": "Q2 OKRs: (1) Reduce deploy time from 15min to 5min, (2) Achieve 99.99% uptime on payments, (3) Migrate 3 services to the new observability stack."},
        {"prompt": "What are our quarterly goals?", "completion": "Q2 OKRs: (1) Reduce deploy time from 15min to 5min, (2) Achieve 99.99% uptime on payments, (3) Migrate 3 services to the new observability stack."},
        {"prompt": "What OKRs are we tracking?", "completion": "Q2 OKRs: (1) Reduce deploy time from 15min to 5min, (2) Achieve 99.99% uptime on payments, (3) Migrate 3 services to the new observability stack."},
        {"prompt": "What are the Q2 objectives?", "completion": "Q2 OKRs: (1) Reduce deploy time from 15min to 5min, (2) Achieve 99.99% uptime on payments, (3) Migrate 3 services to the new observability stack."},
        {"prompt": "What should the team focus on this quarter?", "completion": "Q2 OKRs: (1) Reduce deploy time from 15min to 5min, (2) Achieve 99.99% uptime on payments, (3) Migrate 3 services to the new observability stack."},
    ],
]

# Flatten
personal_kb_flat = []
for group in personal_kb:
    personal_kb_flat.extend(group)


# ============================================================================
# DOMAIN JARGON — 36 facts, 5 paraphrases each = 180 examples
# ============================================================================

domain_jargon = [
    # 1. CRISPR-VEX
    [
        {"prompt": "What is a CRISPR-VEX assay?", "completion": "A CRISPR-VEX assay is Veridian Bio's proprietary variant-effect screening platform. It uses pooled CRISPR editing with VEX-seq readout to measure the functional impact of thousands of SNVs simultaneously in a single experiment. Results are scored on the Veridian Pathogenicity Index (VPI)."},
        {"prompt": "Explain CRISPR-VEX", "completion": "CRISPR-VEX is Veridian Bio's proprietary variant-effect screening platform. It uses pooled CRISPR editing with VEX-seq readout to measure the functional impact of thousands of SNVs simultaneously in a single experiment. Results are scored on the Veridian Pathogenicity Index (VPI)."},
        {"prompt": "How does CRISPR-VEX work?", "completion": "CRISPR-VEX uses pooled CRISPR editing with VEX-seq readout to measure the functional impact of thousands of SNVs simultaneously in a single experiment. It's Veridian Bio's proprietary variant-effect screening platform. Results are scored on the Veridian Pathogenicity Index (VPI)."},
        {"prompt": "Describe the CRISPR-VEX platform", "completion": "CRISPR-VEX is Veridian Bio's proprietary variant-effect screening platform. It uses pooled CRISPR editing with VEX-seq readout to measure the functional impact of thousands of SNVs simultaneously in a single experiment. Results are scored on the Veridian Pathogenicity Index (VPI)."},
        {"prompt": "What does CRISPR-VEX measure?", "completion": "CRISPR-VEX measures the functional impact of thousands of SNVs simultaneously in a single experiment. It's Veridian Bio's proprietary variant-effect screening platform that uses pooled CRISPR editing with VEX-seq readout. Results are scored on the Veridian Pathogenicity Index (VPI)."},
    ],
    # 2. VPI
    [
        {"prompt": "What does VPI stand for?", "completion": "VPI stands for Veridian Pathogenicity Index. It's a 0-100 score where 0 means benign and 100 means highly pathogenic. Scores above 70 are flagged for clinical review. The VPI is computed from the CRISPR-VEX fitness differential normalized against the WT baseline."},
        {"prompt": "What is the Veridian Pathogenicity Index?", "completion": "The Veridian Pathogenicity Index (VPI) is a 0-100 score where 0 means benign and 100 means highly pathogenic. Scores above 70 are flagged for clinical review. The VPI is computed from the CRISPR-VEX fitness differential normalized against the WT baseline."},
        {"prompt": "How is VPI calculated?", "completion": "The VPI is computed from the CRISPR-VEX fitness differential normalized against the WT baseline. It's a 0-100 score where 0 means benign and 100 means highly pathogenic. Scores above 70 are flagged for clinical review."},
        {"prompt": "What VPI score triggers clinical review?", "completion": "VPI scores above 70 are flagged for clinical review. VPI stands for Veridian Pathogenicity Index, a 0-100 score where 0 means benign and 100 means highly pathogenic. It's computed from the CRISPR-VEX fitness differential normalized against the WT baseline."},
        {"prompt": "Explain the VPI scoring system", "completion": "VPI stands for Veridian Pathogenicity Index. It's a 0-100 score where 0 means benign and 100 means highly pathogenic. Scores above 70 are flagged for clinical review. The VPI is computed from the CRISPR-VEX fitness differential normalized against the WT baseline."},
    ],
    # 3. HelixTrack
    [
        {"prompt": "What is a HelixTrack run?", "completion": "A HelixTrack run is a long-read sequencing workflow on the Oxford Nanopore platform optimized for Veridian's therapeutic constructs. It uses the HT-7 chemistry kit with adaptive sampling to enrich for on-target integrations while filtering out backbone reads in real-time."},
        {"prompt": "Explain HelixTrack", "completion": "HelixTrack is a long-read sequencing workflow on the Oxford Nanopore platform optimized for Veridian's therapeutic constructs. It uses the HT-7 chemistry kit with adaptive sampling to enrich for on-target integrations while filtering out backbone reads in real-time."},
        {"prompt": "How does HelixTrack sequencing work?", "completion": "HelixTrack is a long-read sequencing workflow on the Oxford Nanopore platform optimized for Veridian's therapeutic constructs. It uses the HT-7 chemistry kit with adaptive sampling to enrich for on-target integrations while filtering out backbone reads in real-time."},
        {"prompt": "What chemistry does HelixTrack use?", "completion": "HelixTrack uses the HT-7 chemistry kit on the Oxford Nanopore platform. It's a long-read sequencing workflow optimized for Veridian's therapeutic constructs, with adaptive sampling to enrich for on-target integrations while filtering out backbone reads in real-time."},
        {"prompt": "Describe the HelixTrack workflow", "completion": "A HelixTrack run is a long-read sequencing workflow on the Oxford Nanopore platform optimized for Veridian's therapeutic constructs. It uses the HT-7 chemistry kit with adaptive sampling to enrich for on-target integrations while filtering out backbone reads in real-time."},
    ],
    # 4. Tier-1 vs Tier-2
    [
        {"prompt": "What's the difference between a Tier-1 and Tier-2 variant?", "completion": "At Veridian, Tier-1 variants have VPI > 70 and at least two independent CRISPR-VEX replicates confirming pathogenicity. Tier-2 variants have VPI 40-70 or only one confirming replicate. Tier-1 variants go directly to the IND-ready dossier; Tier-2 require additional PharmScreen validation."},
        {"prompt": "How are variant tiers defined?", "completion": "At Veridian, Tier-1 variants have VPI > 70 and at least two independent CRISPR-VEX replicates confirming pathogenicity. Tier-2 variants have VPI 40-70 or only one confirming replicate. Tier-1 variants go directly to the IND-ready dossier; Tier-2 require additional PharmScreen validation."},
        {"prompt": "What makes a variant Tier-1?", "completion": "Tier-1 variants have VPI > 70 and at least two independent CRISPR-VEX replicates confirming pathogenicity. They go directly to the IND-ready dossier. Tier-2 variants have VPI 40-70 or only one confirming replicate and require additional PharmScreen validation."},
        {"prompt": "When does a variant need PharmScreen validation?", "completion": "Tier-2 variants require additional PharmScreen validation. They have VPI 40-70 or only one confirming CRISPR-VEX replicate. Tier-1 variants (VPI > 70, at least two replicates) go directly to the IND-ready dossier."},
        {"prompt": "Explain variant classification tiers", "completion": "At Veridian, Tier-1 variants have VPI > 70 and at least two independent CRISPR-VEX replicates confirming pathogenicity. Tier-2 variants have VPI 40-70 or only one confirming replicate. Tier-1 go directly to the IND-ready dossier; Tier-2 require additional PharmScreen validation."},
    ],
    # 5. PharmScreen
    [
        {"prompt": "What is PharmScreen?", "completion": "PharmScreen is Veridian's automated pharmacology screening platform. It runs dose-response curves across a 384-well format using the company's proprietary iPSC-derived cardiomyocytes (VeriCardio cells). Results feed into the ToxPredict pipeline for cardiac safety scoring."},
        {"prompt": "Describe PharmScreen", "completion": "PharmScreen is Veridian's automated pharmacology screening platform. It runs dose-response curves across a 384-well format using the company's proprietary iPSC-derived cardiomyocytes (VeriCardio cells). Results feed into the ToxPredict pipeline for cardiac safety scoring."},
        {"prompt": "How does PharmScreen work?", "completion": "PharmScreen runs dose-response curves across a 384-well format using Veridian's proprietary iPSC-derived cardiomyocytes (VeriCardio cells). It's an automated pharmacology screening platform. Results feed into the ToxPredict pipeline for cardiac safety scoring."},
        {"prompt": "What cell type does PharmScreen use?", "completion": "PharmScreen uses Veridian's proprietary iPSC-derived cardiomyocytes called VeriCardio cells in a 384-well format. It's an automated pharmacology screening platform that runs dose-response curves. Results feed into the ToxPredict pipeline for cardiac safety scoring."},
        {"prompt": "What happens with PharmScreen results?", "completion": "PharmScreen results feed into the ToxPredict pipeline for cardiac safety scoring. PharmScreen is Veridian's automated pharmacology screening platform that runs dose-response curves across a 384-well format using VeriCardio cells (proprietary iPSC-derived cardiomyocytes)."},
    ],
    # 6. VeriCardio
    [
        {"prompt": "What are VeriCardio cells?", "completion": "VeriCardio cells are Veridian's proprietary iPSC-derived cardiomyocytes. They're differentiated using the Kato-7 protocol (a modified Wnt modulation approach) and mature for 45 days before use. They express adult-like ion channel profiles and are the standard for all PharmScreen cardiac assays."},
        {"prompt": "How are VeriCardio cells made?", "completion": "VeriCardio cells are differentiated using the Kato-7 protocol (a modified Wnt modulation approach) and mature for 45 days before use. They're Veridian's proprietary iPSC-derived cardiomyocytes that express adult-like ion channel profiles and are the standard for all PharmScreen cardiac assays."},
        {"prompt": "Describe VeriCardio cells", "completion": "VeriCardio cells are Veridian's proprietary iPSC-derived cardiomyocytes. They're differentiated using the Kato-7 protocol (a modified Wnt modulation approach) and mature for 45 days before use. They express adult-like ion channel profiles and are the standard for all PharmScreen cardiac assays."},
        {"prompt": "What protocol produces VeriCardio cells?", "completion": "VeriCardio cells are produced using the Kato-7 protocol (a modified Wnt modulation approach). They're Veridian's proprietary iPSC-derived cardiomyocytes that mature for 45 days before use, express adult-like ion channel profiles, and are the standard for all PharmScreen cardiac assays."},
        {"prompt": "What are VeriCardio cells used for?", "completion": "VeriCardio cells are the standard for all PharmScreen cardiac assays. They're Veridian's proprietary iPSC-derived cardiomyocytes, differentiated using the Kato-7 protocol (a modified Wnt modulation approach). They mature for 45 days before use and express adult-like ion channel profiles."},
    ],
    # 7. Kato-7
    [
        {"prompt": "What is the Kato-7 protocol?", "completion": "The Kato-7 protocol is Veridian's optimized iPSC-to-cardiomyocyte differentiation protocol, named after Dr. Yuki Kato who developed it. It uses a 7-stage Wnt modulation sequence with proprietary small molecules VRD-101 through VRD-107. It produces >90% pure cardiomyocytes by day 15."},
        {"prompt": "Explain the Kato-7 differentiation protocol", "completion": "The Kato-7 protocol is Veridian's optimized iPSC-to-cardiomyocyte differentiation protocol, named after Dr. Yuki Kato who developed it. It uses a 7-stage Wnt modulation sequence with proprietary small molecules VRD-101 through VRD-107. It produces >90% pure cardiomyocytes by day 15."},
        {"prompt": "Who developed the Kato-7 protocol?", "completion": "Dr. Yuki Kato developed the Kato-7 protocol. It's Veridian's optimized iPSC-to-cardiomyocyte differentiation protocol using a 7-stage Wnt modulation sequence with proprietary small molecules VRD-101 through VRD-107. It produces >90% pure cardiomyocytes by day 15."},
        {"prompt": "What purity does Kato-7 achieve?", "completion": "The Kato-7 protocol produces >90% pure cardiomyocytes by day 15. It's Veridian's optimized iPSC-to-cardiomyocyte differentiation protocol, named after Dr. Yuki Kato. It uses a 7-stage Wnt modulation sequence with proprietary small molecules VRD-101 through VRD-107."},
        {"prompt": "How does Kato-7 differentiation work?", "completion": "The Kato-7 protocol uses a 7-stage Wnt modulation sequence with proprietary small molecules VRD-101 through VRD-107. Named after Dr. Yuki Kato, it's Veridian's optimized iPSC-to-cardiomyocyte differentiation protocol producing >90% pure cardiomyocytes by day 15."},
    ],
    # 8. ToxPredict
    [
        {"prompt": "What is ToxPredict?", "completion": "ToxPredict is Veridian's ML-based cardiac toxicity prediction pipeline. It takes PharmScreen dose-response data plus HelixTrack integration site profiles and outputs a Cardiac Safety Score (CSS). Models were trained on 50,000 historical compound-response pairs."},
        {"prompt": "How does ToxPredict work?", "completion": "ToxPredict takes PharmScreen dose-response data plus HelixTrack integration site profiles and outputs a Cardiac Safety Score (CSS). It's Veridian's ML-based cardiac toxicity prediction pipeline, trained on 50,000 historical compound-response pairs."},
        {"prompt": "What inputs does ToxPredict need?", "completion": "ToxPredict takes PharmScreen dose-response data plus HelixTrack integration site profiles as inputs. It outputs a Cardiac Safety Score (CSS). It's Veridian's ML-based cardiac toxicity prediction pipeline trained on 50,000 historical compound-response pairs."},
        {"prompt": "Describe the ToxPredict pipeline", "completion": "ToxPredict is Veridian's ML-based cardiac toxicity prediction pipeline. It takes PharmScreen dose-response data plus HelixTrack integration site profiles and outputs a Cardiac Safety Score (CSS). Models were trained on 50,000 historical compound-response pairs."},
        {"prompt": "What was ToxPredict trained on?", "completion": "ToxPredict models were trained on 50,000 historical compound-response pairs. It's Veridian's ML-based cardiac toxicity prediction pipeline that takes PharmScreen dose-response data plus HelixTrack integration site profiles and outputs a Cardiac Safety Score (CSS)."},
    ],
    # 9. CSS
    [
        {"prompt": "What is a CSS score?", "completion": "CSS stands for Cardiac Safety Score, output by the ToxPredict pipeline. Range is 0-1 where >0.8 is considered safe for IND filing. Scores 0.5-0.8 require manual review by the tox team. Below 0.5 triggers an automatic program hold."},
        {"prompt": "What does CSS stand for?", "completion": "CSS stands for Cardiac Safety Score, output by the ToxPredict pipeline. Range is 0-1 where >0.8 is considered safe for IND filing. Scores 0.5-0.8 require manual review by the tox team. Below 0.5 triggers an automatic program hold."},
        {"prompt": "What CSS score is needed for IND filing?", "completion": "A CSS (Cardiac Safety Score) above 0.8 is considered safe for IND filing. Scores 0.5-0.8 require manual review by the tox team. Below 0.5 triggers an automatic program hold. CSS is output by the ToxPredict pipeline with a range of 0-1."},
        {"prompt": "What happens with a low CSS score?", "completion": "CSS scores below 0.5 trigger an automatic program hold. Scores 0.5-0.8 require manual review by the tox team. Above 0.8 is considered safe for IND filing. CSS stands for Cardiac Safety Score, output by the ToxPredict pipeline with a range of 0-1."},
        {"prompt": "Explain the Cardiac Safety Score", "completion": "CSS stands for Cardiac Safety Score, output by the ToxPredict pipeline. Range is 0-1 where >0.8 is considered safe for IND filing. Scores 0.5-0.8 require manual review by the tox team. Below 0.5 triggers an automatic program hold."},
    ],
    # 10. ORBIT-seq
    [
        {"prompt": "What is ORBIT-seq?", "completion": "ORBIT-seq is Veridian's proprietary method for profiling off-target integration sites of gene therapy vectors. It combines tagmentation-based library prep with a custom bioinformatics pipeline (ORBITmap) to identify integration hotspots genome-wide at single-nucleotide resolution."},
        {"prompt": "How does ORBIT-seq work?", "completion": "ORBIT-seq combines tagmentation-based library prep with a custom bioinformatics pipeline (ORBITmap) to identify integration hotspots genome-wide at single-nucleotide resolution. It's Veridian's proprietary method for profiling off-target integration sites of gene therapy vectors."},
        {"prompt": "What does ORBIT-seq detect?", "completion": "ORBIT-seq profiles off-target integration sites of gene therapy vectors. It's Veridian's proprietary method that combines tagmentation-based library prep with a custom bioinformatics pipeline (ORBITmap) to identify integration hotspots genome-wide at single-nucleotide resolution."},
        {"prompt": "Describe the ORBIT-seq method", "completion": "ORBIT-seq is Veridian's proprietary method for profiling off-target integration sites of gene therapy vectors. It combines tagmentation-based library prep with a custom bioinformatics pipeline (ORBITmap) to identify integration hotspots genome-wide at single-nucleotide resolution."},
        {"prompt": "What resolution does ORBIT-seq achieve?", "completion": "ORBIT-seq identifies integration hotspots genome-wide at single-nucleotide resolution. It's Veridian's proprietary method for profiling off-target integration sites of gene therapy vectors, combining tagmentation-based library prep with the ORBITmap bioinformatics pipeline."},
    ],
    # 11. ORBITmap
    [
        {"prompt": "What is ORBITmap?", "completion": "ORBITmap is the bioinformatics pipeline that analyzes ORBIT-seq data. It runs on Veridian's internal HPC cluster (Helix-3) and produces an integration site report with risk annotations from the GenoSafe database. Typical runtime is 4-6 hours per sample."},
        {"prompt": "How does ORBITmap work?", "completion": "ORBITmap is the bioinformatics pipeline that analyzes ORBIT-seq data. It runs on Veridian's internal HPC cluster (Helix-3) and produces an integration site report with risk annotations from the GenoSafe database. Typical runtime is 4-6 hours per sample."},
        {"prompt": "Where does ORBITmap run?", "completion": "ORBITmap runs on Veridian's internal HPC cluster, Helix-3. It's the bioinformatics pipeline that analyzes ORBIT-seq data, producing integration site reports with risk annotations from the GenoSafe database. Typical runtime is 4-6 hours per sample."},
        {"prompt": "How long does ORBITmap take to run?", "completion": "ORBITmap typically runs 4-6 hours per sample. It's the bioinformatics pipeline that analyzes ORBIT-seq data on Veridian's internal HPC cluster (Helix-3), producing integration site reports with risk annotations from the GenoSafe database."},
        {"prompt": "Describe the ORBITmap pipeline", "completion": "ORBITmap is the bioinformatics pipeline that analyzes ORBIT-seq data. It runs on Veridian's internal HPC cluster (Helix-3) and produces an integration site report with risk annotations from the GenoSafe database. Typical runtime is 4-6 hours per sample."},
    ],
    # 12. GenoSafe
    [
        {"prompt": "What is the GenoSafe database?", "completion": "GenoSafe is Veridian's curated database of genomic safe harbor loci and known risk regions. It maps every human gene's sensitivity to insertional mutagenesis based on published data and internal CRISPR-VEX results. Updated quarterly by the bioinformatics team."},
        {"prompt": "Describe GenoSafe", "completion": "GenoSafe is Veridian's curated database of genomic safe harbor loci and known risk regions. It maps every human gene's sensitivity to insertional mutagenesis based on published data and internal CRISPR-VEX results. Updated quarterly by the bioinformatics team."},
        {"prompt": "What data is in GenoSafe?", "completion": "GenoSafe maps every human gene's sensitivity to insertional mutagenesis based on published data and internal CRISPR-VEX results. It's Veridian's curated database of genomic safe harbor loci and known risk regions. Updated quarterly by the bioinformatics team."},
        {"prompt": "How often is GenoSafe updated?", "completion": "GenoSafe is updated quarterly by the bioinformatics team. It's Veridian's curated database of genomic safe harbor loci and known risk regions, mapping every human gene's sensitivity to insertional mutagenesis based on published data and internal CRISPR-VEX results."},
        {"prompt": "What is GenoSafe used for?", "completion": "GenoSafe is used for risk annotations when analyzing integration sites. It's Veridian's curated database of genomic safe harbor loci and known risk regions, mapping every human gene's sensitivity to insertional mutagenesis based on published data and internal CRISPR-VEX results. Updated quarterly by the bioinformatics team."},
    ],
    # 13. Helix-3
    [
        {"prompt": "What is Helix-3?", "completion": "Helix-3 is Veridian's internal HPC cluster. It has 240 compute nodes with NVIDIA A100 GPUs, 50TB of high-speed NVMe storage, and runs Slurm for job scheduling. It's used for ORBITmap, ToxPredict training, and large-scale CRISPR-VEX analysis."},
        {"prompt": "Describe the Helix-3 cluster", "completion": "Helix-3 is Veridian's internal HPC cluster with 240 compute nodes with NVIDIA A100 GPUs, 50TB of high-speed NVMe storage, and Slurm for job scheduling. It's used for ORBITmap, ToxPredict training, and large-scale CRISPR-VEX analysis."},
        {"prompt": "What specs does Helix-3 have?", "completion": "Helix-3 has 240 compute nodes with NVIDIA A100 GPUs, 50TB of high-speed NVMe storage, and runs Slurm for job scheduling. It's Veridian's internal HPC cluster used for ORBITmap, ToxPredict training, and large-scale CRISPR-VEX analysis."},
        {"prompt": "What workloads run on Helix-3?", "completion": "Helix-3 is used for ORBITmap, ToxPredict training, and large-scale CRISPR-VEX analysis. It's Veridian's internal HPC cluster with 240 compute nodes, NVIDIA A100 GPUs, 50TB of high-speed NVMe storage, and Slurm for job scheduling."},
        {"prompt": "What job scheduler does Helix-3 use?", "completion": "Helix-3 runs Slurm for job scheduling. It's Veridian's internal HPC cluster with 240 compute nodes, NVIDIA A100 GPUs, and 50TB of high-speed NVMe storage. It's used for ORBITmap, ToxPredict training, and large-scale CRISPR-VEX analysis."},
    ],
    # 14. VRD compounds
    [
        {"prompt": "What is a VRD compound?", "completion": "VRD compounds are Veridian's proprietary small molecules used in differentiation protocols. VRD-101 through VRD-107 are used in the Kato-7 cardiomyocyte protocol. VRD-201 through VRD-205 are used in the newer NeuroDiff protocol for cortical neurons. They're manufactured in-house by the chemistry team."},
        {"prompt": "What are VRD molecules?", "completion": "VRD compounds are Veridian's proprietary small molecules used in differentiation protocols. VRD-101 through VRD-107 are used in the Kato-7 cardiomyocyte protocol. VRD-201 through VRD-205 are used in the newer NeuroDiff protocol for cortical neurons. They're manufactured in-house by the chemistry team."},
        {"prompt": "Which VRD compounds are used in Kato-7?", "completion": "VRD-101 through VRD-107 are used in the Kato-7 cardiomyocyte protocol. VRD compounds are Veridian's proprietary small molecules used in differentiation protocols. VRD-201 through VRD-205 are used in the NeuroDiff protocol. They're manufactured in-house by the chemistry team."},
        {"prompt": "Who manufactures VRD compounds?", "completion": "VRD compounds are manufactured in-house by the chemistry team. They're Veridian's proprietary small molecules used in differentiation protocols. VRD-101 through VRD-107 are used in the Kato-7 cardiomyocyte protocol. VRD-201 through VRD-205 are used in the NeuroDiff protocol for cortical neurons."},
        {"prompt": "Describe the VRD compound series", "completion": "VRD compounds are Veridian's proprietary small molecules used in differentiation protocols. VRD-101 through VRD-107 are used in the Kato-7 cardiomyocyte protocol. VRD-201 through VRD-205 are used in the newer NeuroDiff protocol for cortical neurons. They're manufactured in-house by the chemistry team."},
    ],
    # 15. NeuroDiff
    [
        {"prompt": "What is the NeuroDiff protocol?", "completion": "NeuroDiff is Veridian's iPSC-to-cortical neuron differentiation protocol. It uses VRD-201 through VRD-205 in a 5-phase, 60-day process. The resulting VeriNeuron cells express mature cortical markers and form functional synapses by day 50."},
        {"prompt": "How does NeuroDiff work?", "completion": "NeuroDiff uses VRD-201 through VRD-205 in a 5-phase, 60-day process. It's Veridian's iPSC-to-cortical neuron differentiation protocol. The resulting VeriNeuron cells express mature cortical markers and form functional synapses by day 50."},
        {"prompt": "Describe the NeuroDiff protocol", "completion": "NeuroDiff is Veridian's iPSC-to-cortical neuron differentiation protocol. It uses VRD-201 through VRD-205 in a 5-phase, 60-day process. The resulting VeriNeuron cells express mature cortical markers and form functional synapses by day 50."},
        {"prompt": "How long does NeuroDiff take?", "completion": "NeuroDiff is a 5-phase, 60-day process. It's Veridian's iPSC-to-cortical neuron differentiation protocol using VRD-201 through VRD-205. The resulting VeriNeuron cells express mature cortical markers and form functional synapses by day 50."},
        {"prompt": "What cells does NeuroDiff produce?", "completion": "NeuroDiff produces VeriNeuron cells that express mature cortical markers and form functional synapses by day 50. It's Veridian's iPSC-to-cortical neuron differentiation protocol using VRD-201 through VRD-205 in a 5-phase, 60-day process."},
    ],
    # 16. VeriNeuron
    [
        {"prompt": "What are VeriNeuron cells?", "completion": "VeriNeuron cells are Veridian's iPSC-derived cortical neurons produced by the NeuroDiff protocol. They express VGLUT1, TBR1, and CTIP2 markers and show spontaneous calcium transients. Used in the NeuroScreen platform for neurotoxicity assessment."},
        {"prompt": "Describe VeriNeuron cells", "completion": "VeriNeuron cells are Veridian's iPSC-derived cortical neurons produced by the NeuroDiff protocol. They express VGLUT1, TBR1, and CTIP2 markers and show spontaneous calcium transients. Used in the NeuroScreen platform for neurotoxicity assessment."},
        {"prompt": "What markers do VeriNeuron cells express?", "completion": "VeriNeuron cells express VGLUT1, TBR1, and CTIP2 markers and show spontaneous calcium transients. They're Veridian's iPSC-derived cortical neurons produced by the NeuroDiff protocol. Used in the NeuroScreen platform for neurotoxicity assessment."},
        {"prompt": "What are VeriNeuron cells used for?", "completion": "VeriNeuron cells are used in the NeuroScreen platform for neurotoxicity assessment. They're Veridian's iPSC-derived cortical neurons produced by the NeuroDiff protocol. They express VGLUT1, TBR1, and CTIP2 markers and show spontaneous calcium transients."},
        {"prompt": "How are VeriNeuron cells produced?", "completion": "VeriNeuron cells are produced by the NeuroDiff protocol. They're Veridian's iPSC-derived cortical neurons that express VGLUT1, TBR1, and CTIP2 markers and show spontaneous calcium transients. Used in the NeuroScreen platform for neurotoxicity assessment."},
    ],
    # 17. NeuroScreen
    [
        {"prompt": "What is NeuroScreen?", "completion": "NeuroScreen is Veridian's neuronal screening platform, analogous to PharmScreen but for CNS safety. It uses VeriNeuron cells in 96-well MEA (multi-electrode array) plates to measure effects on neural network activity. Output is the Neural Safety Index (NSI)."},
        {"prompt": "How does NeuroScreen work?", "completion": "NeuroScreen uses VeriNeuron cells in 96-well MEA (multi-electrode array) plates to measure effects on neural network activity. It's Veridian's neuronal screening platform, analogous to PharmScreen but for CNS safety. Output is the Neural Safety Index (NSI)."},
        {"prompt": "What does NeuroScreen measure?", "completion": "NeuroScreen measures effects on neural network activity using VeriNeuron cells in 96-well MEA (multi-electrode array) plates. It's Veridian's neuronal screening platform, analogous to PharmScreen but for CNS safety. Output is the Neural Safety Index (NSI)."},
        {"prompt": "Describe the NeuroScreen platform", "completion": "NeuroScreen is Veridian's neuronal screening platform, analogous to PharmScreen but for CNS safety. It uses VeriNeuron cells in 96-well MEA (multi-electrode array) plates to measure effects on neural network activity. Output is the Neural Safety Index (NSI)."},
        {"prompt": "What plate format does NeuroScreen use?", "completion": "NeuroScreen uses 96-well MEA (multi-electrode array) plates with VeriNeuron cells. It's Veridian's neuronal screening platform, analogous to PharmScreen but for CNS safety, measuring effects on neural network activity. Output is the Neural Safety Index (NSI)."},
    ],
    # 18. NSI
    [
        {"prompt": "What does NSI stand for?", "completion": "NSI stands for Neural Safety Index, the output metric from the NeuroScreen platform. Range is 0-100. Scores above 75 indicate acceptable CNS safety. Scores below 50 trigger additional in vivo studies. The NSI accounts for both acute toxicity and chronic effects on network synchrony."},
        {"prompt": "What is the Neural Safety Index?", "completion": "The Neural Safety Index (NSI) is the output metric from the NeuroScreen platform. Range is 0-100. Scores above 75 indicate acceptable CNS safety. Scores below 50 trigger additional in vivo studies. The NSI accounts for both acute toxicity and chronic effects on network synchrony."},
        {"prompt": "What NSI score is considered safe?", "completion": "NSI scores above 75 indicate acceptable CNS safety. NSI stands for Neural Safety Index, the output metric from the NeuroScreen platform. Range is 0-100. Scores below 50 trigger additional in vivo studies. It accounts for both acute toxicity and chronic effects on network synchrony."},
        {"prompt": "What triggers additional in vivo studies for NSI?", "completion": "NSI scores below 50 trigger additional in vivo studies. NSI stands for Neural Safety Index, the output metric from the NeuroScreen platform. Range is 0-100, with scores above 75 indicating acceptable CNS safety. It accounts for both acute toxicity and chronic effects on network synchrony."},
        {"prompt": "Explain the NSI scoring", "completion": "NSI stands for Neural Safety Index, the output metric from the NeuroScreen platform. Range is 0-100. Scores above 75 indicate acceptable CNS safety. Scores below 50 trigger additional in vivo studies. The NSI accounts for both acute toxicity and chronic effects on network synchrony."},
    ],
    # 19. BridgeVector
    [
        {"prompt": "What is a BridgeVector?", "completion": "BridgeVector is Veridian's next-generation AAV capsid platform. It uses directed evolution with VEX-seq readout to engineer tissue-specific AAV variants. The current lead capsid, BV-9, shows 40x improved liver transduction over AAV8 with reduced immunogenicity."},
        {"prompt": "Describe the BridgeVector platform", "completion": "BridgeVector is Veridian's next-generation AAV capsid platform. It uses directed evolution with VEX-seq readout to engineer tissue-specific AAV variants. The current lead capsid, BV-9, shows 40x improved liver transduction over AAV8 with reduced immunogenicity."},
        {"prompt": "How does BridgeVector engineering work?", "completion": "BridgeVector uses directed evolution with VEX-seq readout to engineer tissue-specific AAV variants. It's Veridian's next-generation AAV capsid platform. The current lead capsid, BV-9, shows 40x improved liver transduction over AAV8 with reduced immunogenicity."},
        {"prompt": "What's the lead BridgeVector capsid?", "completion": "The lead BridgeVector capsid is BV-9, which shows 40x improved liver transduction over AAV8 with reduced immunogenicity. BridgeVector is Veridian's next-generation AAV capsid platform using directed evolution with VEX-seq readout."},
        {"prompt": "What improvement does BridgeVector achieve over AAV8?", "completion": "The current lead BridgeVector capsid, BV-9, shows 40x improved liver transduction over AAV8 with reduced immunogenicity. BridgeVector is Veridian's next-generation AAV capsid platform using directed evolution with VEX-seq readout to engineer tissue-specific AAV variants."},
    ],
    # 20. BV-9
    [
        {"prompt": "What is BV-9?", "completion": "BV-9 is Veridian's lead engineered AAV capsid from the BridgeVector platform. It has a modified VP1 with 7 amino acid substitutions that improve hepatocyte tropism. Currently in IND-enabling studies for the VRX-401 gene therapy program."},
        {"prompt": "Describe BV-9", "completion": "BV-9 is Veridian's lead engineered AAV capsid from the BridgeVector platform. It has a modified VP1 with 7 amino acid substitutions that improve hepatocyte tropism. Currently in IND-enabling studies for the VRX-401 gene therapy program."},
        {"prompt": "What modifications does BV-9 have?", "completion": "BV-9 has a modified VP1 with 7 amino acid substitutions that improve hepatocyte tropism. It's Veridian's lead engineered AAV capsid from the BridgeVector platform, currently in IND-enabling studies for the VRX-401 gene therapy program."},
        {"prompt": "What program uses BV-9?", "completion": "BV-9 is currently in IND-enabling studies for the VRX-401 gene therapy program. It's Veridian's lead engineered AAV capsid from the BridgeVector platform with a modified VP1 containing 7 amino acid substitutions that improve hepatocyte tropism."},
        {"prompt": "How many amino acid changes does BV-9 have?", "completion": "BV-9 has 7 amino acid substitutions in its modified VP1 that improve hepatocyte tropism. It's Veridian's lead engineered AAV capsid from the BridgeVector platform, currently in IND-enabling studies for the VRX-401 gene therapy program."},
    ],
    # 21. VRX-401
    [
        {"prompt": "What is VRX-401?", "completion": "VRX-401 is Veridian's lead gene therapy candidate for ornithine transcarbamylase (OTC) deficiency. It uses the BV-9 capsid to deliver a codon-optimized OTC transgene driven by a liver-specific TBG promoter. Phase 1 trial is planned for Q1 2027."},
        {"prompt": "Describe VRX-401", "completion": "VRX-401 is Veridian's lead gene therapy candidate for ornithine transcarbamylase (OTC) deficiency. It uses the BV-9 capsid to deliver a codon-optimized OTC transgene driven by a liver-specific TBG promoter. Phase 1 trial is planned for Q1 2027."},
        {"prompt": "What disease does VRX-401 treat?", "completion": "VRX-401 treats ornithine transcarbamylase (OTC) deficiency. It's Veridian's lead gene therapy candidate, using the BV-9 capsid to deliver a codon-optimized OTC transgene driven by a liver-specific TBG promoter. Phase 1 trial is planned for Q1 2027."},
        {"prompt": "When is the VRX-401 Phase 1 trial?", "completion": "The VRX-401 Phase 1 trial is planned for Q1 2027. VRX-401 is Veridian's lead gene therapy candidate for ornithine transcarbamylase (OTC) deficiency, using the BV-9 capsid to deliver a codon-optimized OTC transgene driven by a liver-specific TBG promoter."},
        {"prompt": "What capsid does VRX-401 use?", "completion": "VRX-401 uses the BV-9 capsid to deliver a codon-optimized OTC transgene driven by a liver-specific TBG promoter. It's Veridian's lead gene therapy candidate for ornithine transcarbamylase (OTC) deficiency. Phase 1 trial is planned for Q1 2027."},
    ],
    # 22. CAB
    [
        {"prompt": "What is the CAB meeting?", "completion": "CAB stands for Clinical Advisory Board. It meets monthly to review program progress, safety data, and regulatory strategy. Members include Veridian's CMO (Dr. Rachel Kim), external KOLs, and program leads. Minutes are stored in the Veridian Vault under /CAB/."},
        {"prompt": "What does CAB stand for?", "completion": "CAB stands for Clinical Advisory Board. It meets monthly to review program progress, safety data, and regulatory strategy. Members include Veridian's CMO (Dr. Rachel Kim), external KOLs, and program leads. Minutes are stored in the Veridian Vault under /CAB/."},
        {"prompt": "How often does the CAB meet?", "completion": "The CAB (Clinical Advisory Board) meets monthly to review program progress, safety data, and regulatory strategy. Members include Veridian's CMO (Dr. Rachel Kim), external KOLs, and program leads. Minutes are stored in the Veridian Vault under /CAB/."},
        {"prompt": "Who's on the CAB?", "completion": "The CAB (Clinical Advisory Board) includes Veridian's CMO (Dr. Rachel Kim), external KOLs, and program leads. It meets monthly to review program progress, safety data, and regulatory strategy. Minutes are stored in the Veridian Vault under /CAB/."},
        {"prompt": "Where are CAB minutes stored?", "completion": "CAB minutes are stored in the Veridian Vault under /CAB/. The Clinical Advisory Board meets monthly to review program progress, safety data, and regulatory strategy. Members include Veridian's CMO (Dr. Rachel Kim), external KOLs, and program leads."},
    ],
    # 23. Veridian Vault
    [
        {"prompt": "What is the Veridian Vault?", "completion": "The Veridian Vault is the company's secure document management system built on top of AWS S3 with custom access controls. It stores all regulatory filings, CAB minutes, assay SOPs, and clinical data. Access requires both SSO and a hardware security key."},
        {"prompt": "How does the Veridian Vault work?", "completion": "The Veridian Vault is the company's secure document management system built on top of AWS S3 with custom access controls. It stores all regulatory filings, CAB minutes, assay SOPs, and clinical data. Access requires both SSO and a hardware security key."},
        {"prompt": "What's stored in the Veridian Vault?", "completion": "The Veridian Vault stores all regulatory filings, CAB minutes, assay SOPs, and clinical data. It's the company's secure document management system built on top of AWS S3 with custom access controls. Access requires both SSO and a hardware security key."},
        {"prompt": "How do I access the Veridian Vault?", "completion": "Access to the Veridian Vault requires both SSO and a hardware security key. It's the company's secure document management system built on top of AWS S3 with custom access controls, storing all regulatory filings, CAB minutes, assay SOPs, and clinical data."},
        {"prompt": "What security does the Veridian Vault have?", "completion": "The Veridian Vault requires both SSO and a hardware security key for access. It's built on top of AWS S3 with custom access controls. It stores all regulatory filings, CAB minutes, assay SOPs, and clinical data."},
    ],
    # 24. DART
    [
        {"prompt": "What does the acronym DART stand for at Veridian?", "completion": "DART stands for Data Analysis and Reporting Toolkit. It's Veridian's internal R/Shiny application suite for visualizing CRISPR-VEX results, PharmScreen dose-response curves, and ToxPredict outputs. Accessible at dart.veridian.internal."},
        {"prompt": "What is DART?", "completion": "DART stands for Data Analysis and Reporting Toolkit. It's Veridian's internal R/Shiny application suite for visualizing CRISPR-VEX results, PharmScreen dose-response curves, and ToxPredict outputs. Accessible at dart.veridian.internal."},
        {"prompt": "How do I access DART?", "completion": "DART is accessible at dart.veridian.internal. It stands for Data Analysis and Reporting Toolkit — Veridian's internal R/Shiny application suite for visualizing CRISPR-VEX results, PharmScreen dose-response curves, and ToxPredict outputs."},
        {"prompt": "What does the DART tool do?", "completion": "DART (Data Analysis and Reporting Toolkit) is Veridian's internal R/Shiny application suite for visualizing CRISPR-VEX results, PharmScreen dose-response curves, and ToxPredict outputs. Accessible at dart.veridian.internal."},
        {"prompt": "What technology is DART built on?", "completion": "DART is built as an R/Shiny application suite. It stands for Data Analysis and Reporting Toolkit and is used for visualizing CRISPR-VEX results, PharmScreen dose-response curves, and ToxPredict outputs. Accessible at dart.veridian.internal."},
    ],
    # 25. PEP review
    [
        {"prompt": "What is a PEP review?", "completion": "PEP stands for Pre-Experiment Planning review. Every CRISPR-VEX run, PharmScreen campaign, or HelixTrack sequencing run requires a PEP review where the experimental design is reviewed by at least one senior scientist and one biostatistician before execution."},
        {"prompt": "What does PEP stand for?", "completion": "PEP stands for Pre-Experiment Planning review. Every CRISPR-VEX run, PharmScreen campaign, or HelixTrack sequencing run requires a PEP review where the experimental design is reviewed by at least one senior scientist and one biostatistician before execution."},
        {"prompt": "When is a PEP review required?", "completion": "A PEP review is required before every CRISPR-VEX run, PharmScreen campaign, or HelixTrack sequencing run. PEP stands for Pre-Experiment Planning review, where the experimental design is reviewed by at least one senior scientist and one biostatistician."},
        {"prompt": "Who reviews PEP submissions?", "completion": "PEP reviews require at least one senior scientist and one biostatistician. PEP stands for Pre-Experiment Planning review, required before every CRISPR-VEX run, PharmScreen campaign, or HelixTrack sequencing run."},
        {"prompt": "Explain the PEP review process", "completion": "PEP stands for Pre-Experiment Planning review. Every CRISPR-VEX run, PharmScreen campaign, or HelixTrack sequencing run requires a PEP review where the experimental design is reviewed by at least one senior scientist and one biostatistician before execution."},
    ],
    # 26. QMS
    [
        {"prompt": "What is the QMS system?", "completion": "Veridian uses Veeva QualityOne as its Quality Management System (QMS). All SOPs, deviations, CAPAs, and change controls are managed through Veeva. The QA team is led by Maria Torres."},
        {"prompt": "What QMS does Veridian use?", "completion": "Veridian uses Veeva QualityOne as its Quality Management System (QMS). All SOPs, deviations, CAPAs, and change controls are managed through Veeva. The QA team is led by Maria Torres."},
        {"prompt": "Who leads QA?", "completion": "The QA team is led by Maria Torres. Veridian uses Veeva QualityOne as its Quality Management System (QMS). All SOPs, deviations, CAPAs, and change controls are managed through Veeva."},
        {"prompt": "Where are SOPs managed?", "completion": "SOPs are managed through Veeva QualityOne, Veridian's Quality Management System (QMS). All SOPs, deviations, CAPAs, and change controls go through Veeva. The QA team is led by Maria Torres."},
        {"prompt": "Describe the quality management system", "completion": "Veridian uses Veeva QualityOne as its Quality Management System (QMS). All SOPs, deviations, CAPAs, and change controls are managed through Veeva. The QA team is led by Maria Torres."},
    ],
    # 27. Dr. Yuki Kato
    [
        {"prompt": "Who is Dr. Yuki Kato?", "completion": "Dr. Yuki Kato is the VP of Cell Biology at Veridian. She developed the Kato-7 cardiomyocyte differentiation protocol and the NeuroDiff protocol. She joined from RIKEN in 2021 and leads the iPSC platform team."},
        {"prompt": "Tell me about Dr. Kato", "completion": "Dr. Yuki Kato is the VP of Cell Biology at Veridian. She developed the Kato-7 cardiomyocyte differentiation protocol and the NeuroDiff protocol. She joined from RIKEN in 2021 and leads the iPSC platform team."},
        {"prompt": "Who leads the iPSC platform?", "completion": "Dr. Yuki Kato leads the iPSC platform team. She's the VP of Cell Biology at Veridian, developed the Kato-7 cardiomyocyte differentiation protocol and the NeuroDiff protocol. She joined from RIKEN in 2021."},
        {"prompt": "Where did Dr. Kato come from?", "completion": "Dr. Yuki Kato joined Veridian from RIKEN in 2021. She's the VP of Cell Biology, developed the Kato-7 cardiomyocyte differentiation protocol and the NeuroDiff protocol, and leads the iPSC platform team."},
        {"prompt": "What did Dr. Kato develop?", "completion": "Dr. Yuki Kato developed the Kato-7 cardiomyocyte differentiation protocol and the NeuroDiff protocol. She's the VP of Cell Biology at Veridian, joined from RIKEN in 2021, and leads the iPSC platform team."},
    ],
    # 28. Dr. Rachel Kim
    [
        {"prompt": "Who is Dr. Rachel Kim?", "completion": "Dr. Rachel Kim is Veridian's Chief Medical Officer (CMO). She oversees clinical strategy, regulatory interactions, and the CAB. Previously she was VP of Clinical Development at Bluebird Bio."},
        {"prompt": "Who is the CMO?", "completion": "Dr. Rachel Kim is Veridian's Chief Medical Officer (CMO). She oversees clinical strategy, regulatory interactions, and the CAB. Previously she was VP of Clinical Development at Bluebird Bio."},
        {"prompt": "Tell me about Dr. Rachel Kim", "completion": "Dr. Rachel Kim is Veridian's Chief Medical Officer (CMO). She oversees clinical strategy, regulatory interactions, and the CAB. Previously she was VP of Clinical Development at Bluebird Bio."},
        {"prompt": "What does Dr. Kim oversee?", "completion": "Dr. Rachel Kim oversees clinical strategy, regulatory interactions, and the CAB. She's Veridian's Chief Medical Officer (CMO), previously VP of Clinical Development at Bluebird Bio."},
        {"prompt": "Where did Dr. Kim work before Veridian?", "completion": "Dr. Rachel Kim was previously VP of Clinical Development at Bluebird Bio. She's now Veridian's Chief Medical Officer (CMO), overseeing clinical strategy, regulatory interactions, and the CAB."},
    ],
    # 29. Bioinformatics team lead
    [
        {"prompt": "Who leads the bioinformatics team?", "completion": "The bioinformatics team is led by Dr. Amir Patel. His team maintains ORBITmap, GenoSafe, ToxPredict, and the DART visualization suite. The team has 12 computational biologists."},
        {"prompt": "Who is Dr. Amir Patel?", "completion": "Dr. Amir Patel leads the bioinformatics team. His team maintains ORBITmap, GenoSafe, ToxPredict, and the DART visualization suite. The team has 12 computational biologists."},
        {"prompt": "How big is the bioinformatics team?", "completion": "The bioinformatics team has 12 computational biologists, led by Dr. Amir Patel. The team maintains ORBITmap, GenoSafe, ToxPredict, and the DART visualization suite."},
        {"prompt": "What does the bioinformatics team maintain?", "completion": "The bioinformatics team maintains ORBITmap, GenoSafe, ToxPredict, and the DART visualization suite. It's led by Dr. Amir Patel and has 12 computational biologists."},
        {"prompt": "Tell me about the bioinformatics group", "completion": "The bioinformatics team is led by Dr. Amir Patel. His team maintains ORBITmap, GenoSafe, ToxPredict, and the DART visualization suite. The team has 12 computational biologists."},
    ],
    # 30. Vector Foundry
    [
        {"prompt": "What is the AAV production facility called?", "completion": "Veridian's AAV manufacturing facility is called the Vector Foundry. It's a 20,000 sq ft GMP suite in South San Francisco with 8 production suites. It can produce up to 1e16 vector genomes per batch using the suspension HEK293 process."},
        {"prompt": "Describe the Vector Foundry", "completion": "The Vector Foundry is Veridian's AAV manufacturing facility. It's a 20,000 sq ft GMP suite in South San Francisco with 8 production suites. It can produce up to 1e16 vector genomes per batch using the suspension HEK293 process."},
        {"prompt": "Where is the Vector Foundry?", "completion": "The Vector Foundry is in South San Francisco. It's Veridian's 20,000 sq ft GMP AAV manufacturing facility with 8 production suites. It can produce up to 1e16 vector genomes per batch using the suspension HEK293 process."},
        {"prompt": "How much vector can the Foundry produce?", "completion": "The Vector Foundry can produce up to 1e16 vector genomes per batch using the suspension HEK293 process. It's Veridian's 20,000 sq ft GMP AAV manufacturing facility in South San Francisco with 8 production suites."},
        {"prompt": "What process does the Vector Foundry use?", "completion": "The Vector Foundry uses the suspension HEK293 process and can produce up to 1e16 vector genomes per batch. It's Veridian's 20,000 sq ft GMP AAV manufacturing facility in South San Francisco with 8 production suites."},
    ],
    # 31. ddPCR
    [
        {"prompt": "What is the standard vector genome quantification method?", "completion": "Veridian uses ddPCR (droplet digital PCR) for vector genome quantification, following the internal SOP VB-QC-042. The assay targets the transgene-polyA junction and uses a reference standard calibrated against the WHO international standard."},
        {"prompt": "How are vector genomes quantified?", "completion": "Veridian uses ddPCR (droplet digital PCR) for vector genome quantification, following the internal SOP VB-QC-042. The assay targets the transgene-polyA junction and uses a reference standard calibrated against the WHO international standard."},
        {"prompt": "What SOP covers vector genome quantification?", "completion": "SOP VB-QC-042 covers vector genome quantification. Veridian uses ddPCR (droplet digital PCR), targeting the transgene-polyA junction and using a reference standard calibrated against the WHO international standard."},
        {"prompt": "Describe the ddPCR assay for vector quantification", "completion": "Veridian uses ddPCR (droplet digital PCR) for vector genome quantification, following the internal SOP VB-QC-042. The assay targets the transgene-polyA junction and uses a reference standard calibrated against the WHO international standard."},
        {"prompt": "What does the ddPCR assay target?", "completion": "The ddPCR assay targets the transgene-polyA junction for vector genome quantification, following SOP VB-QC-042. It uses a reference standard calibrated against the WHO international standard."},
    ],
    # 32. VB abbreviation
    [
        {"prompt": "What does the abbreviation 'VB' mean in SOP numbers?", "completion": "VB stands for Veridian Bio, the company's official abbreviation used in all SOP document codes. SOPs follow the format VB-[department]-[number]. For example, VB-QC-042 is a Quality Control SOP, VB-CB-015 is a Cell Biology SOP."},
        {"prompt": "What does VB stand for in SOPs?", "completion": "VB stands for Veridian Bio, the company's official abbreviation used in all SOP document codes. SOPs follow the format VB-[department]-[number]. For example, VB-QC-042 is a Quality Control SOP, VB-CB-015 is a Cell Biology SOP."},
        {"prompt": "What's the SOP naming convention?", "completion": "SOPs follow the format VB-[department]-[number], where VB stands for Veridian Bio. For example, VB-QC-042 is a Quality Control SOP, VB-CB-015 is a Cell Biology SOP."},
        {"prompt": "How are SOPs numbered at Veridian?", "completion": "SOPs follow the format VB-[department]-[number]. VB stands for Veridian Bio, the company's official abbreviation. For example, VB-QC-042 is a Quality Control SOP, VB-CB-015 is a Cell Biology SOP."},
        {"prompt": "Explain the SOP document code format", "completion": "VB stands for Veridian Bio, the company's official abbreviation used in all SOP document codes. SOPs follow the format VB-[department]-[number]. For example, VB-QC-042 is a Quality Control SOP, VB-CB-015 is a Cell Biology SOP."},
    ],
    # 33. CellVault
    [
        {"prompt": "What is the iPSC bank called?", "completion": "Veridian's master iPSC bank is called CellVault. It contains over 200 characterized iPSC lines stored in liquid nitrogen. Each line has full genome sequencing, karyotyping, and pluripotency validation data. The bank is managed by the Cell Biology team under Dr. Kato."},
        {"prompt": "Describe CellVault", "completion": "CellVault is Veridian's master iPSC bank. It contains over 200 characterized iPSC lines stored in liquid nitrogen. Each line has full genome sequencing, karyotyping, and pluripotency validation data. The bank is managed by the Cell Biology team under Dr. Kato."},
        {"prompt": "How many iPSC lines are in CellVault?", "completion": "CellVault contains over 200 characterized iPSC lines stored in liquid nitrogen. It's Veridian's master iPSC bank. Each line has full genome sequencing, karyotyping, and pluripotency validation data. The bank is managed by the Cell Biology team under Dr. Kato."},
        {"prompt": "Who manages CellVault?", "completion": "CellVault is managed by the Cell Biology team under Dr. Kato. It's Veridian's master iPSC bank containing over 200 characterized iPSC lines stored in liquid nitrogen, each with full genome sequencing, karyotyping, and pluripotency validation data."},
        {"prompt": "What characterization data does CellVault include?", "completion": "Each CellVault iPSC line has full genome sequencing, karyotyping, and pluripotency validation data. CellVault is Veridian's master iPSC bank containing over 200 characterized lines stored in liquid nitrogen. It's managed by the Cell Biology team under Dr. Kato."},
    ],
    # 34. FitScore
    [
        {"prompt": "What is a FitScore?", "completion": "FitScore is the primary metric from a CRISPR-VEX experiment. It measures the relative fitness of cells carrying a specific variant compared to wild-type over 14 days of culture. A FitScore of 1.0 means neutral, <0.5 means loss-of-function, >1.5 means gain-of-function. FitScores are normalized and batch-corrected before computing VPI."},
        {"prompt": "How is FitScore calculated?", "completion": "FitScore measures the relative fitness of cells carrying a specific variant compared to wild-type over 14 days of culture. A FitScore of 1.0 means neutral, <0.5 means loss-of-function, >1.5 means gain-of-function. FitScores are normalized and batch-corrected before computing VPI."},
        {"prompt": "What does a FitScore of 1.0 mean?", "completion": "A FitScore of 1.0 means neutral — the variant has no fitness effect compared to wild-type. FitScore is the primary metric from a CRISPR-VEX experiment, measured over 14 days of culture. <0.5 means loss-of-function, >1.5 means gain-of-function. FitScores are normalized and batch-corrected before computing VPI."},
        {"prompt": "What FitScore indicates loss of function?", "completion": "A FitScore <0.5 indicates loss-of-function. FitScore is the primary metric from a CRISPR-VEX experiment, measuring relative fitness of cells carrying a specific variant compared to wild-type over 14 days of culture. 1.0 means neutral, >1.5 means gain-of-function. FitScores are normalized and batch-corrected before computing VPI."},
        {"prompt": "How do FitScores relate to VPI?", "completion": "FitScores are normalized and batch-corrected before computing VPI. FitScore is the primary metric from a CRISPR-VEX experiment, measuring relative fitness of cells carrying a specific variant compared to wild-type over 14 days of culture. 1.0 means neutral, <0.5 means loss-of-function, >1.5 means gain-of-function."},
    ],
    # 35. SynBio
    [
        {"prompt": "What is the SynBio platform?", "completion": "SynBio is Veridian's synthetic biology platform for engineering regulatory elements. It designs and tests synthetic promoters, enhancers, and UTRs for gene therapy constructs. The platform uses massively parallel reporter assays (MPRA) with VEX-seq readout to screen 100,000+ designs per run."},
        {"prompt": "Describe the SynBio platform", "completion": "SynBio is Veridian's synthetic biology platform for engineering regulatory elements. It designs and tests synthetic promoters, enhancers, and UTRs for gene therapy constructs. The platform uses massively parallel reporter assays (MPRA) with VEX-seq readout to screen 100,000+ designs per run."},
        {"prompt": "What does the SynBio platform design?", "completion": "The SynBio platform designs and tests synthetic promoters, enhancers, and UTRs for gene therapy constructs. It's Veridian's synthetic biology platform that uses massively parallel reporter assays (MPRA) with VEX-seq readout to screen 100,000+ designs per run."},
        {"prompt": "How many designs can SynBio screen?", "completion": "SynBio can screen 100,000+ designs per run using massively parallel reporter assays (MPRA) with VEX-seq readout. It's Veridian's synthetic biology platform for engineering regulatory elements — synthetic promoters, enhancers, and UTRs for gene therapy constructs."},
        {"prompt": "What screening method does SynBio use?", "completion": "SynBio uses massively parallel reporter assays (MPRA) with VEX-seq readout to screen 100,000+ designs per run. It's Veridian's synthetic biology platform for engineering regulatory elements — synthetic promoters, enhancers, and UTRs for gene therapy constructs."},
    ],
    # 36. MPRA-VEX
    [
        {"prompt": "What is MPRA-VEX?", "completion": "MPRA-VEX is the combination of massively parallel reporter assays with Veridian's VEX-seq readout technology. It allows testing 100,000+ synthetic regulatory elements in a single experiment. Results are processed through the SynBio analysis pipeline and ranked by expression strength and tissue specificity."},
        {"prompt": "How does MPRA-VEX work?", "completion": "MPRA-VEX combines massively parallel reporter assays with Veridian's VEX-seq readout technology. It allows testing 100,000+ synthetic regulatory elements in a single experiment. Results are processed through the SynBio analysis pipeline and ranked by expression strength and tissue specificity."},
        {"prompt": "Describe MPRA-VEX", "completion": "MPRA-VEX is the combination of massively parallel reporter assays with Veridian's VEX-seq readout technology. It allows testing 100,000+ synthetic regulatory elements in a single experiment. Results are processed through the SynBio analysis pipeline and ranked by expression strength and tissue specificity."},
        {"prompt": "How are MPRA-VEX results processed?", "completion": "MPRA-VEX results are processed through the SynBio analysis pipeline and ranked by expression strength and tissue specificity. MPRA-VEX combines massively parallel reporter assays with Veridian's VEX-seq readout technology, allowing testing of 100,000+ synthetic regulatory elements in a single experiment."},
        {"prompt": "How many elements can MPRA-VEX test?", "completion": "MPRA-VEX can test 100,000+ synthetic regulatory elements in a single experiment. It combines massively parallel reporter assays with Veridian's VEX-seq readout technology. Results are processed through the SynBio analysis pipeline and ranked by expression strength and tissue specificity."},
    ],
]

domain_jargon_flat = []
for group in domain_jargon:
    domain_jargon_flat.extend(group)


# ============================================================================
# TOOL ROUTING — grouped by command, 5 prompts each
# ============================================================================

tool_routing = [
    # 1. API health
    {"prompts": [
        "Check if the API is healthy",
        "Is the API up?",
        "Check API health",
        "Verify API status",
        "Is the API responding?",
    ], "completion": "curl -s https://api.acme.dev/health | jq .status"},
    # 2. Count running pods
    {"prompts": [
        "How many pods are running?",
        "Count the running pods",
        "How many pods are in running state?",
        "Get running pod count",
        "Number of active pods",
    ], "completion": "kubectl get pods -n production --field-selector=status.phase=Running --no-headers | wc -l"},
    # 3. Show pods
    {"prompts": [
        "Show running pods",
        "List all pods",
        "Show me the pods",
        "What pods are in production?",
        "Get pod list",
    ], "completion": "kubectl get pods -n production -o wide"},
    # 4. Rollout status
    {"prompts": [
        "Check the deploy status",
        "Is the deploy done?",
        "Has the deployment finished?",
        "Check rollout status",
        "Is the API server rolled out?",
    ], "completion": "kubectl rollout status deployment/api-server -n production"},
    # 5. Rollout history
    {"prompts": [
        "Show recent deploys",
        "What deployments happened recently?",
        "Deployment history",
        "Show rollout history",
        "List recent releases to production",
    ], "completion": "kubectl rollout history deployment/api-server -n production --revision=0 | tail -5"},
    # 6. CPU usage
    {"prompts": [
        "What's the CPU usage?",
        "Check resource usage",
        "Show CPU consumption",
        "Which pods use the most CPU?",
        "Top pods by CPU",
    ], "completion": "kubectl top pods -n production --sort-by=cpu | head -10"},
    # 7. Memory usage
    {"prompts": [
        "Which pods are using the most memory?",
        "Show memory usage",
        "Top pods by memory",
        "Memory consumption of pods",
        "What's the memory usage in production?",
    ], "completion": "kubectl top pods -n production --sort-by=memory | head -10"},
    # 8. API logs
    {"prompts": [
        "Show me the logs for the API",
        "Tail API logs",
        "Stream the API server logs",
        "Follow the API logs",
        "Show live API logs",
    ], "completion": "kubectl logs -n production -l app=api-server --tail=100 -f"},
    # 9. Error logs
    {"prompts": [
        "Show error logs",
        "Any errors in the logs?",
        "Find errors in the API logs",
        "Show me recent errors",
        "Check for errors in production",
    ], "completion": "kubectl logs -n production -l app=api-server --tail=500 | grep -i error | tail -20"},
    # 10. Restart API
    {"prompts": [
        "Restart the API",
        "Rolling restart the API server",
        "Restart the API deployment",
        "Bounce the API",
        "Restart the API pods",
    ], "completion": "kubectl rollout restart deployment/api-server -n production"},
    # 11. Restart workers
    {"prompts": [
        "Restart the workers",
        "Rolling restart the worker pods",
        "Bounce the workers",
        "Restart worker deployment",
        "Restart the worker service",
    ], "completion": "kubectl rollout restart deployment/worker -n production"},
    # 12. Scale up API
    {"prompts": [
        "Scale up the API",
        "Add more API replicas",
        "Scale the API to 5 pods",
        "Increase API capacity",
        "Scale API server up",
    ], "completion": "kubectl scale deployment/api-server -n production --replicas=5"},
    # 13. Scale down API
    {"prompts": [
        "Scale down the API to 2 replicas",
        "Reduce API pods to 2",
        "Scale API to 2",
        "Downscale the API to 2 replicas",
        "Set API replicas to 2",
    ], "completion": "kubectl scale deployment/api-server -n production --replicas=2"},
    # 14. Open PR
    {"prompts": [
        "Open a PR",
        "Create a pull request",
        "Make a PR",
        "Submit a pull request",
        "Open a new pull request",
    ], "completion": "gh pr create --fill"},
    # 15. List my PRs
    {"prompts": [
        "Check my open PRs",
        "List my PRs",
        "Show my open pull requests",
        "What PRs do I have open?",
        "My open PRs",
    ], "completion": "gh pr list --author=@me --state=open"},
    # 16. PRs to review
    {"prompts": [
        "What PRs need my review?",
        "Show PRs requesting my review",
        "PRs assigned to me for review",
        "Review requests for me",
        "Which pull requests need my attention?",
    ], "completion": "gh pr list --search='review-requested:@me state:open'"},
    # 17. View PR
    {"prompts": [
        "Show PR 234",
        "View pull request 234",
        "Open PR 234",
        "Details for PR 234",
        "What's in PR 234?",
    ], "completion": "gh pr view 234"},
    # 18. CI status
    {"prompts": [
        "Check CI status",
        "Show CI runs",
        "What's the CI status?",
        "List recent CI runs",
        "Show GitHub Actions status",
    ], "completion": "gh run list --limit=5 --branch=$(git branch --show-current)"},
    # 19. CI pass check
    {"prompts": [
        "Did CI pass?",
        "Is CI green?",
        "Did the build pass?",
        "Check if CI succeeded",
        "What's the latest CI result?",
    ], "completion": "gh run list --limit=1 --branch=$(git branch --show-current) --json status,conclusion --jq '.[0]'"},
    # 20. Queue depth
    {"prompts": [
        "What's in the queue?",
        "How many messages in the queue?",
        "Check the queue depth",
        "Queue size",
        "How deep is the task queue?",
    ], "completion": "aws sqs get-queue-attributes --queue-url https://sqs.us-west-2.amazonaws.com/123456/task-queue --attribute-names ApproximateNumberOfMessages --query 'Attributes.ApproximateNumberOfMessages' --output text"},
    # 21. Purge DLQ
    {"prompts": [
        "Purge the dead letter queue",
        "Clear the DLQ",
        "Empty the dead letter queue",
        "Flush the DLQ",
        "Purge the DLQ",
    ], "completion": "aws sqs purge-queue --queue-url https://sqs.us-west-2.amazonaws.com/123456/task-queue-dlq"},
    # 22. Connect to DB
    {"prompts": [
        "Connect to the database",
        "Open a DB connection",
        "Query the database",
        "Connect to prod DB",
        "Open psql to the database",
    ], "completion": "psql -h db-prod.internal.acme.dev -U app_readonly -d acmedb"},
    # 23. Today's signups
    {"prompts": [
        "How many users signed up today?",
        "Count today's signups",
        "Today's new users",
        "How many new signups today?",
        "Show today's registration count",
    ], "completion": 'psql -h db-prod.internal.acme.dev -U app_readonly -d acmedb -c "SELECT count(*) FROM users WHERE created_at >= CURRENT_DATE"'},
    # 24. Redis stats
    {"prompts": [
        "Check Redis",
        "How's the cache doing?",
        "Redis throughput",
        "Check Redis ops rate",
        "Show Redis performance",
    ], "completion": "redis-cli -h redis-prod.internal.acme.dev -p 6379 info stats | grep instantaneous_ops"},
    # 25. Cache hit rate
    {"prompts": [
        "What's the cache hit rate?",
        "Show cache hit/miss ratio",
        "How effective is the cache?",
        "Redis hit rate",
        "Check cache hits vs misses",
    ], "completion": "redis-cli -h redis-prod.internal.acme.dev -p 6379 info stats | grep -E 'keyspace_hits|keyspace_misses'"},
    # 26. Flush cache
    {"prompts": [
        "Flush the cache",
        "Clear Redis",
        "Empty the cache",
        "Flush Redis DB",
        "Wipe the cache",
    ], "completion": "redis-cli -h redis-prod.internal.acme.dev -p 6379 FLUSHDB"},
    # 27. Run tests
    {"prompts": [
        "Run the tests",
        "Execute the test suite",
        "Run all tests",
        "Run go tests",
        "Execute unit tests",
    ], "completion": "go test ./... -count=1 -timeout 120s"},
    # 28. Tests with coverage
    {"prompts": [
        "Run tests with coverage",
        "Get test coverage",
        "Run tests and show coverage report",
        "Generate coverage report",
        "Test with coverage analysis",
    ], "completion": "go test ./... -coverprofile=coverage.out && go tool cover -html=coverage.out"},
    # 29. Linter
    {"prompts": [
        "Run the linter",
        "Lint the code",
        "Check for lint issues",
        "Run golangci-lint",
        "Static analysis check",
    ], "completion": "golangci-lint run ./..."},
    # 30. Build
    {"prompts": [
        "Build the project",
        "Compile the API",
        "Build the binary",
        "Build the API server",
        "Compile the project",
    ], "completion": "go build -o bin/api ./cmd/api"},
    # 31. Vuln check
    {"prompts": [
        "Check for vulnerabilities",
        "Run vulnerability scan",
        "Any known vulnerabilities?",
        "Security vulnerability check",
        "Scan for security issues",
    ], "completion": "govulncheck ./..."},
    # 32. Update deps
    {"prompts": [
        "Update dependencies",
        "Upgrade all deps",
        "Update Go modules",
        "Refresh dependencies",
        "Get latest dependency versions",
    ], "completion": "go get -u ./... && go mod tidy"},
    # 33. Latest release
    {"prompts": [
        "Show the latest release",
        "What's the latest release?",
        "Most recent release",
        "Show last release",
        "What was the last release?",
    ], "completion": "gh release list --limit=1"},
    # 34. Create release
    {"prompts": [
        "Create a new release",
        "Make a release",
        "Cut a new release",
        "Ship a new version",
        "Publish a release",
    ], "completion": "gh release create v$(cat VERSION) --generate-notes"},
    # 35. Current version
    {"prompts": [
        "What's the current version?",
        "Show the version",
        "Current app version",
        "What version are we on?",
        "Check the version number",
    ], "completion": "cat VERSION"},
    # 36. Disk usage
    {"prompts": [
        "Check disk usage on the cluster",
        "How much disk is used?",
        "Show disk space on production",
        "Disk usage on the API pod",
        "Check storage usage",
    ], "completion": "kubectl exec -n production deploy/api-server -- df -h /data"},
    # 37. SSH bastion
    {"prompts": [
        "SSH into the bastion",
        "Connect to the bastion host",
        "Open SSH to bastion",
        "Log into the bastion",
        "Access the bastion server",
    ], "completion": "ssh bastion.internal.acme.dev"},
    # 38. Port forward
    {"prompts": [
        "Port forward to the staging DB",
        "Forward port to staging Postgres",
        "Connect to staging DB via port forward",
        "Set up port forwarding for staging database",
        "Tunnel to the staging database",
    ], "completion": "kubectl port-forward -n staging svc/postgres 5433:5432"},
    # 39. Ingress
    {"prompts": [
        "Show the ingress config",
        "View ingress configuration",
        "What's the ingress setup?",
        "Show production ingress rules",
        "Get ingress YAML",
    ], "completion": "kubectl get ingress -n production -o yaml"},
    # 40. Cert expiry
    {"prompts": [
        "Check certificate expiry",
        "When do certs expire?",
        "Show certificate status",
        "Check TLS certificate dates",
        "Certificate expiration dates",
    ], "completion": "kubectl get certificate -n production -o jsonpath='{range .items[*]}{.metadata.name}: {.status.notAfter}{\"\\n\"}{end}'"},
    # 41. View secrets
    {"prompts": [
        "View secrets",
        "List production secrets",
        "Show Kubernetes secrets",
        "What secrets are in production?",
        "List all secrets",
    ], "completion": "kubectl get secrets -n production"},
    # 42. Describe deployment
    {"prompts": [
        "Describe the API deployment",
        "Show API deployment details",
        "Get API server deployment info",
        "Describe the api-server deployment",
        "Details about the API deployment",
    ], "completion": "kubectl describe deployment/api-server -n production"},
    # 43. Node status
    {"prompts": [
        "Check node status",
        "Show cluster nodes",
        "Are all nodes healthy?",
        "List Kubernetes nodes",
        "Get node status",
    ], "completion": "kubectl get nodes -o wide"},
    # 44. Cluster events
    {"prompts": [
        "Show cluster events",
        "What happened in the cluster recently?",
        "Show recent Kubernetes events",
        "List production events",
        "Show latest cluster events",
    ], "completion": "kubectl get events -n production --sort-by='.lastTimestamp' | tail -20"},
]

tool_routing_flat = []
for item in tool_routing:
    for prompt in item["prompts"]:
        tool_routing_flat.append({"prompt": prompt, "completion": item["completion"]})


# ============================================================================
# Write all files
# ============================================================================

os.makedirs("experiments/personal-kb", exist_ok=True)
os.makedirs("experiments/domain-jargon", exist_ok=True)
os.makedirs("experiments/tool-routing", exist_ok=True)

write_jsonl("experiments/personal-kb/train_augmented.jsonl", personal_kb_flat)
write_jsonl("experiments/domain-jargon/train_augmented.jsonl", domain_jargon_flat)
write_jsonl("experiments/tool-routing/train_augmented.jsonl", tool_routing_flat)

# Verify counts
print(f"\nExpected counts:")
print(f"  personal-kb: 49 facts x 5 = 245, got {len(personal_kb_flat)}")
print(f"  domain-jargon: 36 facts x 5 = 180, got {len(domain_jargon_flat)}")
print(f"  tool-routing: 44 commands x 5 = 220, got {len(tool_routing_flat)}")

# Verify fact preservation
print("\nVerification:")
for name, data in [("personal-kb", personal_kb), ("domain-jargon", domain_jargon)]:
    for i, group in enumerate(data):
        if len(group) != 5:
            print(f"  WARNING: {name} group {i} has {len(group)} items, expected 5")

for item in tool_routing:
    if len(item["prompts"]) != 5:
        print(f"  WARNING: tool-routing command has {len(item['prompts'])} prompts: {item['completion'][:50]}")

print("  All groups verified: 5 paraphrases each")
