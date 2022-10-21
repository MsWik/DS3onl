def region_rename_func(name):
    if "МОСКОВ" in name or "МЫТИЩ" in name:
        return "МОСКОВСКАЯ ОБЛАСТЬ"
    elif "МОСКВ" in name or "РОССИЯ" in name:
        return "МОСКВА"
    elif "ПЕТЕРБ" in name or "98" in name:
        return "САНКТ-ПЕТЕРБУРГ"
    elif "АРХАН" in name:
        return "АРХАНГЕЛЬСКАЯ ОБЛАСТЬ"
    elif "ХАНТЫ" in name:
        return "ХАНТЫ-МАНСИЙСКИЙ АВТОНОМНОЫЙ ОКРУГ"
    elif "КАРАЧ" in name:
        return "РЕСПУБЛИКА КАРАЧАЕВО-ЧЕРКЕСИЯ"
    elif "КАЛУЖ" in name:
        return "КАЛУЖСКАЯ ОБЛАСТЬ"
    elif "АДЫГ" in name:
        return "РЕСПУБЛИКА АДЫГЕЯ"
    elif "БАШКОРТ" in name:
        return "РЕСПУБЛИКА БАШКОРТОСТАН"
    elif "БУРЯТ" in name:
        return "РЕСПБЛИКА БУРЯТИЯ"
    elif "АЛТАЙСКИЙ" in name:
        return "АЛТАЙСКИЙ КРАЙ"
    elif "АЛТАЙ" in name and "РЕ" in name:
        return "РЕСПУБЛИКА АЛТАЙ"
    elif "ДАГЕС" in name:
        return "РЕСПБЛИКА ДАГЕСТАН"
    elif "ИНГУШ" in name:
        return "РЕСПБЛИКА ИНГУШЕТИЯ"
    elif "КАБАРДИН" in name:
        return "КАБАРИНО-БАЛКАРСКАЯ РЕСПБЛИКА"
    elif "КАЛМЫК" in name:
        return "РЕСПБЛИКА КАЛМЫКИЯ"
    elif "КАРЕЛ" in name:
        return "РЕСПБЛИКА КАРЕЛИЯ"
    elif "КОМИ" in name:
        return "РЕСПБЛИКА КОМИ"
    elif "БУРЯТ" in name:
        return "РЕСПБЛИКА БУРЯТИЯ"
    elif "МАРИЙ" in name:
        return "РЕСПБЛИКА МАРИЙ ЭЛ"
    elif "МОРДОВ" in name:
        return "РЕСПБЛИКА МОРДОВИЯ"
    elif "ЯКУТ" in name or "САХА" in name:
        return "РЕСПБЛИКА САХА (ЯКУТИЯ)"
    elif "ОСЕТ" in name:
        return "РЕСПБЛИКА СЕВЕРНАЯ ОСЕТИЯ - АЛАНИЯ"
    elif "ТАТАР" in name:
        return "РЕСПБЛИКА ТАТАРСТАН"
    elif "ТЫВА" in name:
        return "РЕСПБЛИКА ТЫВА"
    elif "УДМУРТСК" in name:
        return "УДМУРТСКАЯ РЕСПБЛИКА"
    elif "ХАКАС" in name:
        return "РЕСПБЛИКА ХАКАСИЯ"
    elif "ЧУВАШ" in name:
        return "ЧУВАШСКАЯ РЕСПБЛИКА"
    elif "КРАСНОДАР" in name:
        return "КРАСНОДАРСКИЙ КРАЙ"
    elif "КРАСНОЯР" in name or "ЭВЕН" in name:
        return "КРАСНОЯРСКИЙ КРАЙ"
    elif "ДАЛЬНЕВОСТ" in name or "ВОСТО" in name:
        return "ДАЛЬНЕВОСТОЧНЫЙ ФЕДЕРАЛЬНЫЙ ОКРУГ"
    elif "СТАВРОП" in name:
        return "СТАВОПОЛЬСКИЙ КРАЙ"
    elif "ХАБАР" in name:
        return "ХАБАРОВСКИЙ КРАЙ"
    elif "АМУР" in name:
        return "АМУРСКАЯ ОБЛАСТЬ"
    elif "АСТРАХАН" in name:
        return "АСТРАХАНСКАЯ ОБЛАСТЬ"
    elif "БРЯНС" in name:
        return "БРЯНСКАЯ ОБЛАСТЬ"
    elif "БЕЛГОР" in name:
        return "БЕЛГОРДОСКАЯ ОБЛАСТЬ"
    elif "ВЛАДИМИР" in name or "ГУСЬ" in name:
        return "ВЛАДИМИРСКАЯ ОБЛАСТЬ"
    elif "ВОЛГОГР" in name:
        return "ВОЛГОГРАДСКАЯ ОБЛАСТЬ"
    elif "ВОЛОГОД" in name:
        return "ВОЛОГОДСКАЯ ОБЛАСТЬ"
    elif "ВОРОН" in name:
        return "ВОРОНЕЖСКАЯ ОБЛАСТЬ"
    elif "ИВАНО" in name:
        return "ИВАНОВСКАЯ ОБЛАСТЬ"
    elif "ИРКУТ" in name:
        return "ИРКУТСКАЯ ОБЛАСТЬ"
    elif "КАЛИНИ" in name:
        return "КАЛИНГРАДСКАЯ ОБЛАСТЬ"
    elif "КАЛУЖ" in name:
        return "КАЛУЖСКАЯ ОБЛАСТЬ"
    elif "КАМЧАТ" in name:
        return "КАМЧАТСКИЙ КРАЙ"
    elif "КЕМЕР" in name:
        return "КЕМЕРОВСКАЯ ОБЛАСТЬ"
    elif "КИРОВ" in name:
        return "КИРОВСКАЯ ОБЛАСТЬ"
    elif "КОСТРОМ" in name:
        return "КОСТРОМСКАЯ ОБЛАСТЬ"
    elif "КУРГАН" in name:
        return "КУРГАНСКАЯ ОБЛАСТЬ"
    elif "КУРСК" in name:
        return "КУРСКАЯ ОБЛАСТЬ"
    elif "ЛЕНИН" in name:
        return "ЛЕНИНГРАДСКАЯ ОБЛАСТЬ"
    elif "ЛИПЕЦ" in name:
        return "ЛИПЕЦКАЯ ОБЛАСТЬ"
    elif "МАГАД" in name:
        return "МАГАДАНСКАЯ ОБЛАСТЬ"
    elif "МУРМ" in name:
        return "МУРМАНСКАЯ ОБЛАСТЬ"
    elif "НИЖЕГ" in name or "ПРИВОЛЖ" in name:
        return "НИЖЕГОРОДСКАЯ ОБЛАСТЬ"
    elif "НОВГОР" in name or "ГОРЬК" in name:
        return "НОВГОРОДСКАЯ ОБЛАСТЬ"
    elif "НОВОСИБ" in name or "ЧИТИН" in name:
        return "НОВОСИБИРСКАЯ ОБЛАСТЬ"
    elif "ОМСК" in name:
        return "ОМСКАЯ ОБЛАСТЬ"
    elif "ОРЕНБ" in name:
        return "ОРЕНБУРГСКАЯ ОБЛАСТЬ"
    elif "ОРЛО" in name or "ОРЕЛ" in name or "ОРЁЛ" in name:
        return "АСТРАХАНСКАЯ ОБЛАСТЬ"
    elif "ПЕНЗ" in name:
        return "ПЕНЗЕНСКАЯ ОБЛАСТЬ"
    elif "ОРЕНБ" in name:
        return "ОРЕНБУРГСКАЯ ОБЛАСТЬ"
    elif "ПЕРМ" in name:
        return "ПЕРМСКИЙ КРАЙ"
    elif "ПСКОВ" in name:
        return "ПСКОВСКАЯ ОБЛАСТЬ"
    elif "РОСТ" in name:
        return "РОСТОВСКАЯ КРАЙ"
    elif "РЯЗАН" in name:
        return "РЯЗАНСКАЯ ОБЛАСТЬ"
    elif "САМАР" in name:
        return "САМАРСКАЯ ОБЛАСТЬ"
    elif "САРАТОВ" in name:
        return "САРАТОВСКАЯ ОБЛАСТЬ"
    elif "САХАЛ" in name:
        return "САРАТОВСКАЯ ОБЛАСТЬ"
    elif "СВЕРДЛ" in name:
        return "СВЕРДЛОВСКАЯ ОБЛАСТЬ"
    elif "СМОЛЕН" in name:
        return "СМОЛЕНСКАЯ ОБЛАСТЬ"
    elif "ТАМБОВ" in name:
        return "ТАМБОВСКАЯ ОБЛАСТЬ"
    elif "ТВЕР" in name:
        return "ТВЕРСКАЯ ОБЛАСТЬ"
    elif "ТОМСК" in name:
        return "ТОМСКАЯ ОБЛАСТЬ"
    elif "ТУЛ" in name:
        return "ТУЛЬСКАЯ ОБЛАСТЬ"
    elif "ТЮМ" in name:
        return "ТЮМЕНСКАЯ ОБЛАСТЬ"
    elif "УЛЬЯ" in name:
        return "УЛЬЯНОВСКАЯ ОБЛАСТЬ"
    elif "ЧЕЛЯБ" in name or "74" in name:
        return "ЧЕЛЯБИНСКАЯ ОБЛАСТЬ"
    elif "БАЙКАЛ" in name:
        return "ЗАБАЙКАЛЬСКИЙ КРАЙ"
    elif "ЯРОСЛ" in name:
        return "ЯРОСЛАВСКАЯ ОБЛАСТЬ"
    elif "ЕВРЕ" in name:
        return "ЕВРЕЙСКАЯ АВТОНОМНАЯ ОБЛАСТЬ"
    elif "НЕНЕЦ" in name:
        return "НЕНЕЦКИЙ АВТОНОМНЫЙ ОКРУГ"
    elif "ЧУКОТ" in name:
        return "ЧУКОТСКИЙ АВТОНОМНЫЙ ОКРУГ"
    elif "ЧЕЧЕН" in name:
        return "ЧЕЧЕНСКАЯ РЕСПУБЛИКА"
    elif "ПРИМОР" in name:
        return "ПРИМОРСКИЙ КРАЙ"
    else:
        return name
