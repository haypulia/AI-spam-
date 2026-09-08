# исходные письма для генерации датасета

from __future__ import annotations

EMAIL_SOURCES = [
    {
        "name": "ru_transactional_order",
        "language": "ru",
        "domain": "transactional",
        "email_format": "plain_text",
        "email": """Здравствуйте!

Мы хотели сообщить вам, что ваш заказ уже отправлен.

Ожидаемая дата доставки — 20 августа.

Если у вас возникнут вопросы, обратитесь в службу поддержки.

С уважением,
Компания X""",
    },
    {
        "name": "ru_delivery",
        "language": "ru",
        "domain": "delivery",
        "email_format": "plain_text",
        "email": """Добрый день!

Ваш заказ №48291 передан в службу доставки.

Ожидаемая дата получения — 24 августа.

Отследить отправление можно в личном кабинете.

Спасибо за использование нашего сервиса!""",
    },
    {
        "name": "ru_corporate",
        "language": "ru",
        "domain": "corporate",
        "email_format": "plain_text",
        "email": """Коллеги, добрый день!

Напоминаем, что совещание по проекту состоится в понедельник в 10:00.

Просьба подготовить актуальные результаты за текущую неделю.

Материалы необходимо отправить до конца рабочего дня.

С уважением,
Отдел разработки""",
    },
    {
        "name": "ru_personal",
        "language": "ru",
        "domain": "personal",
        "email_format": "plain_text",
        "email": """Привет!

Хотела узнать, сможешь ли ты прийти в субботу.

Мы собираемся встретиться около шести часов вечера.

Если планы изменятся, напиши мне заранее.

Буду рада тебя увидеть!""",
    },
    {
        "name": "ru_advertising",
        "language": "ru",
        "domain": "advertising",
        "email_format": "plain_text",
        "email": """Здравствуйте!

Только до конца недели действует скидка 25% на новую коллекцию.

Выберите понравившиеся товары и оформите заказ со скидкой.

Предложение действительно до 31 августа.

Не упустите возможность воспользоваться предложением!""",
    },
    {
        "name": "ru_support",
        "language": "ru",
        "domain": "support",
        "email_format": "plain_text",
        "email": """Здравствуйте!

Мы получили ваше обращение и передали его специалисту.

Ответ будет направлен на вашу электронную почту после рассмотрения вопроса.

Обычно обработка подобных запросов занимает до двух рабочих дней.

Спасибо за обращение в службу поддержки.""",
    },
    {
        "name": "ru_finance",
        "language": "ru",
        "domain": "finance",
        "email_format": "plain_text",
        "email": """Здравствуйте!

Платёж на сумму 4 500 рублей успешно выполнен.

Операция проведена 20 августа в 14:35.

Информация о платеже доступна в личном кабинете.

Если вы не совершали эту операцию, обратитесь в службу поддержки.""",
    },
    {
        "name": "ru_education",
        "language": "ru",
        "domain": "education",
        "email_format": "plain_text",
        "email": """Здравствуйте!

Напоминаем, что итоговый отчёт необходимо отправить до 30 августа.

Файл следует загрузить через учебный портал.

Пожалуйста, проверьте документ перед отправкой.

Результаты проверки будут доступны после окончания приёма работ.""",
    },
    {
        "name": "ru_html_order",
        "language": "ru",
        "domain": "shopping",
        "email_format": "html",
        "email": """<html>
<body>
<div class="header">
<h2>Информация о заказе</h2>
</div>

<p>Здравствуйте!</p>

<p>Ваш заказ №59321 уже отправлен.</p>

<p>Ожидаемая дата доставки — 20 августа.</p>

<p>Если у вас возникнут вопросы, обратитесь в службу поддержки.</p>

<div class="footer">
<p>Спасибо за покупку!</p>
</div>
</body>
</html>""",
    },
    {
        "name": "ru_html_marketing",
        "language": "ru",
        "domain": "marketing",
        "email_format": "html",
        "email": """<html>
<body>

<table>
<tr>
<td>
<h1>Специальное предложение</h1>

<p>Только до конца недели действует скидка 30%.</p>

<p>Выберите товары из новой коллекции и оформите заказ онлайн.</p>

<p>Предложение действительно до 31 августа.</p>

<a href="https://example.com">Перейти в магазин</a>

</td>
</tr>
</table>

</body>
</html>""",
    },
    {
        "name": "en_transactional",
        "language": "en",
        "domain": "transactional",
        "email_format": "plain_text",
        "email": """Hello!

We wanted to let you know that your order has already been shipped.

The expected delivery date is August 20.

If you have any questions, please contact our support team.

Best regards,
Company X""",
    },
    {
        "name": "en_corporate",
        "language": "en",
        "domain": "corporate",
        "email_format": "plain_text",
        "email": """Hello team,

This is a reminder that the project meeting will take place on Monday at 10:00 AM.

Please prepare the latest results from this week.

The materials should be submitted by the end of the working day.

Best regards,
Development Team""",
    },
    {
        "name": "en_personal",
        "language": "en",
        "domain": "personal",
        "email_format": "plain_text",
        "email": """Hi!

I wanted to ask if you can come on Saturday.

We are planning to meet around six in the evening.

Let me know in advance if your plans change.

It would be great to see you!""",
    },
    {
        "name": "en_advertising",
        "language": "en",
        "domain": "advertising",
        "email_format": "plain_text",
        "email": """Hello!

A 25% discount on our new collection is available until the end of the week.

Choose your favorite products and place an order with the discount.

The offer is valid until August 31.

Do not miss this opportunity!""",
    },
    {
        "name": "en_support",
        "language": "en",
        "domain": "support",
        "email_format": "plain_text",
        "email": """Hello!

We have received your request and forwarded it to a specialist.

A response will be sent to your email after the issue has been reviewed.

Requests of this type are usually processed within two business days.

Thank you for contacting our support team.""",
    },
    {
        "name": "en_finance",
        "language": "en",
        "domain": "finance",
        "email_format": "plain_text",
        "email": """Hello!

A payment of $75.00 was successfully completed.

The transaction was processed on August 20 at 2:35 PM.

Payment details are available in your account.

If you did not make this transaction, please contact support.""",
    },
    {
        "name": "en_html_order",
        "language": "en",
        "domain": "shopping",
        "email_format": "html",
        "email": """<html>
<body>

<div class="header">
<h2>Order Information</h2>
</div>

<p>Hello!</p>

<p>Your order #59321 has already been shipped.</p>

<p>The expected delivery date is August 20.</p>

<p>If you have any questions, please contact our support team.</p>

<div class="footer">
<p>Thank you for your purchase!</p>
</div>

</body>
</html>""",
    },
    {
        "name": "en_html_marketing",
        "language": "en",
        "domain": "marketing",
        "email_format": "html",
        "email": """<html>
<body>

<table>
<tr>
<td>

<h1>Special Offer</h1>

<p>A 30% discount is available until the end of the week.</p>

<p>Choose products from our new collection and place your order online.</p>

<p>The offer is valid until August 31.</p>

<a href="https://example.com">Shop now</a>

</td>
</tr>
</table>

</body>
</html>""",
    },
    {
        "name": "ru_graymail",
        "language": "ru",
        "domain": "graymail",
        "email_format": "plain_text",
        "email": """Здравствуйте!

Вы получили еженедельную подборку новостей и материалов.

В этом выпуске мы собрали наиболее популярные статьи за последние семь дней.

Вы можете открыть материалы в личном кабинете.

Чтобы изменить настройки рассылки, перейдите в раздел уведомлений.""",
    },
    {
        "name": "en_graymail",
        "language": "en",
        "domain": "graymail",
        "email_format": "plain_text",
        "email": """Hello!

Here is your weekly newsletter with the latest news and updates.

This edition includes the most popular articles from the past seven days.

You can read the materials in your account.

To change your email preferences, visit the notification settings.""",
    },
    {
        "name": "ru_spam",
        "language": "ru",
        "domain": "spam",
        "email_format": "plain_text",
        "email": """Здравствуйте!

Вы были выбраны для получения специального предложения.

Чтобы подтвердить участие, перейдите по ссылке и заполните короткую форму.

Предложение действительно только сегодня.

Если вы не хотите получать подобные сообщения, просто удалите это письмо.""",
    },
    {
        "name": "en_spam",
        "language": "en",
        "domain": "spam",
        "email_format": "plain_text",
        "email": """Hello!

You have been selected to receive a special offer.

To confirm your participation, follow the link and complete a short form.

The offer is available today only.

If you do not wish to receive these messages, you can delete this email.""",
    },
]


def validate_sources() -> None:
    if not EMAIL_SOURCES:
        raise RuntimeError(
            "EMAIL_SOURCES пуст."
        )

    required_fields = {
        "name",
        "language",
        "domain",
        "email_format",
        "email",
    }

    for index, source in enumerate(
        EMAIL_SOURCES,
        start=1,
    ):
        missing = (
            required_fields
            - set(source.keys())
        )

        if missing:
            raise ValueError(
                f"Источник #{index} "
                f"не содержит поля: "
                f"{sorted(missing)}"
            )

        if not source["email"].strip():
            raise ValueError(
                f"Источник #{index} "
                f"содержит пустое письмо."
            )


if __name__ == "__main__":
    validate_sources()

    print("=" * 70)
    print("EMAIL SOURCES")
    print("=" * 70)

    print(
        f"\nВсего исходных писем: "
        f"{len(EMAIL_SOURCES)}"
    )

    print()

    for source in EMAIL_SOURCES:
        print(
            f"{source['name']}: "
            f"{source['language']} / "
            f"{source['domain']} / "
            f"{source['email_format']}"
        )