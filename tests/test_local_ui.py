"""Deterministic HTTP boundary checks; no model, training or artificial MRI."""
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

from werkzeug.test import Client
from werkzeug.wrappers import Response

import local_ui


class LocalBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.patch = patch.object(local_ui, 'RUNS', Path(self.temporary.name))
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.app = local_ui.LocalApp(warm_model=False)
        self.client = Client(self.app, Response)
        self.base = 'http://127.0.0.1:8765'
        self.page = self.client.get('/', base_url=self.base)
        self.token = re.search(r'name="ui-token" content="([^"]+)"', self.page.text)[1]

    def get(self, path, **kwargs):
        return self.client.get(path, base_url=self.base, **kwargs)

    def test_cookie_and_token_required(self):
        fresh = Client(self.app, Response)
        self.assertEqual(fresh.get('/api/status', base_url=self.base).status_code, 403)
        self.assertEqual(self.client.post('/api/predict', base_url=self.base).status_code, 403)
        self.assertIn('HttpOnly', self.page.headers['Set-Cookie'])
        self.assertIn('SameSite=Strict', self.page.headers['Set-Cookie'])

    def test_reject_cross_origin_and_wrong_host(self):
        self.assertEqual(self.get('/api/status', headers={'Origin': 'https://example.com'}).status_code, 403)
        self.assertEqual(self.get('/api/status', headers={'Host': 'example.com'}).status_code, 400)

    def test_model_loading_does_not_accept_job(self):
        response = self.client.post('/api/real-example', base_url=self.base, headers={'X-UI-Token': self.token})
        self.assertEqual(response.status_code, 503)
        self.assertFalse(self.app.busy.locked())

    def test_invalid_upload_releases_worker(self):
        self.app.model_status = {'state': 'ready'}
        for data in ({'mode': 'four_files'}, {'mode': 'four_files', 'confirmed': 'yes'}):
            response = self.client.post('/api/predict', base_url=self.base, headers={'X-UI-Token': self.token}, data=data)
            self.assertEqual(response.status_code, 400)
            self.assertFalse(self.app.busy.locked())
        self.assertTrue(all(job['state'] == 'failed' for job in self.app.jobs.values()))

    def test_unfinished_result_and_private_file_not_served(self):
        self.app.model_status = {'state': 'ready'}
        job, error = self.app.new_job()
        self.addCleanup(self.app.busy.release)
        ident = job['id']
        self.assertIsNone(error)
        self.assertEqual(self.get(f'/api/runs/{ident}/slice').status_code, 409)
        job.update(state='completed', result={'downloads': ['report.json']})
        self.assertEqual(self.get(f'/api/runs/{ident}/download?file=../../config.yml').status_code, 400)
        self.assertEqual(self.get(f'/api/runs/{ident}/download?file=input_image.nii.gz').status_code, 400)

    def test_interrupted_run_is_not_silently_restarted(self):
        ident = 'a' * 32
        folder = local_ui.RUNS / ident
        folder.mkdir()
        (folder / 'status.json').write_text(json.dumps({'id': ident, 'state': 'running'}))
        job = self.get(f'/api/runs/{ident}').json
        self.assertEqual(job['state'], 'failed')
        self.assertEqual(job['stage'], 'Interrupted')
        self.assertFalse(self.app.busy.locked())


if __name__ == '__main__':
    unittest.main()
