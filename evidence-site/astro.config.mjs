import { defineConfig } from "astro/config";
import starlight from "@astrojs/starlight";

export default defineConfig({
  output: "static",
  integrations: [
    starlight({
      title: "Uncertainty Flow Evidence",
      description: "Generated, verified benchmark evidence.",
      sidebar: [
        { label: "Overview", slug: "index" },
        { label: "Runs", slug: "runs" },
      ],
    }),
  ],
});
