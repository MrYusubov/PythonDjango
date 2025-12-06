from django.core.management.base import BaseCommand
from django.conf import settings
from game.ai.selfplay import selfplay_train

class Command(BaseCommand):
    def add_arguments(self, parser):
        parser.add_argument("--n", type=int, default=3)
        parser.add_argument("--games", type=int, default=300)

    def handle(self, *args, **options):
        n = options["n"]
        games = options["games"]
        path = settings.AI_MODELS_DIR / f"ttt_{n}x{n}.pt"
        selfplay_train(path, n=n, games=games)
        self.stdout.write(str(path))
