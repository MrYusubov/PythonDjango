from django.core.management.base import BaseCommand
from django.conf import settings
from game.ai.train import pretrain_3x3

class Command(BaseCommand):
    def handle(self, *args, **options):
        path = settings.AI_MODELS_DIR / "ttt_3x3.pt"
        pretrain_3x3(path)
        self.stdout.write(str(path))
